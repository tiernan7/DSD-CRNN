"""
Synthetic data generation for model training and testing.

This module provides functions to generate synthetic data that can be used
for training and testing the models when real data is not available.
"""

from dataclasses import dataclass
import itertools
import itertools
from operator import le
from unicodedata import name
from networkx import sigma
from numpy.linalg import lstsq
import numpy as np
from numpy.linalg import matrix_rank, svd
from scipy.linalg import null_space
import torch
from  itertools import combinations_with_replacement
from scipy.integrate import solve_ivp
import multiprocessing
from multiprocessing import Pool
import math
@dataclass
class Partition:
    inputs: list[str]
    hidden: list[str]
    output: str = "S_out"

    @property
    def species(self) -> list[str]:
        return self.inputs + self.hidden + [self.output]
        
    @property
    def input_indices(self) -> list[int]:
        return [self.species.index(name) for name in self.inputs]

    @property
    def output_indices(self) -> list[int]:
        return [self.species.index(self.output)]
    
    
@dataclass(frozen=True)
class Reaction:
    reactants: tuple[int, ...]
    products: tuple[int, ...]
    


class SyntheticDataGenerator:
    def __init__(
        self,
        mechanism: str,
        noise_level,
        t_end=40.0,
        n_timepoints=50,
        eps = 1e-12,
        seed=None,
    ):
        
        self.rng = np.random.default_rng(seed)
        self.mechanism = mechanism
        self.noise_level = noise_level
        self.t_end = t_end
        self.n_timepoints = n_timepoints
        self.eps = eps
        self.seed = seed

        (   self.composition_matrix,
            self.partition,
            self.reactions,
        ) = self.mech_to_f(mechanism)()

       
    
    def mech_to_f(self, mechanism: str):
        """
        Convert mechanism string to function.
        Currently only supports hardcoded mechanisms.
        """
        if mechanism == "one_step":
            return self.setup_hardcoded_single_step
        elif mechanism == "one_step_1latent":
            return self.setup_hardcoded_1latent_example
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")


    def _build_stoichiometric_matrix(self):
        """
        Build the stoichiometric matrix S from the reactions.
        Returns:
            np.ndarray: Stoichiometric matrix S of shape (n_species, n_reactions)
        """
        n_species = len(self.partition.species)
        n_rxns = len(self.reactions)
        S = np.zeros((n_species, n_rxns), dtype=float)

        for j, rxn in enumerate(self.reactions):
            for i in rxn.reactants:
                S[i, j] += -1
            for i in rxn.products:
                S[i, j] += 1

        return S

    def _rate_vector(self, k, c, alpha, c_floor):
        # c: (N_ps,)
        # phi: (N_r,)
        # alpha: (N_ps, N_r)
        phi = np.log(k)
        c_safe = np.maximum(c, c_floor)          # for log only
        logc = np.log(c_safe)                    # (N_ps,)
        # log rates: (N_r,) = phi + logc^T alpha
        log_r = phi + (logc @ alpha)             # (N_r,)
        r = np.exp(log_r)
        return r

    def _rate_vector_true(self, c, k):
        """
        Mass-action rates for the *true* CRN (no extra species).
        c: (N_s,) concentrations for partition.species
        k: (N_r,) rate constants
        """
        c = np.asarray(c, dtype=float)
        c = np.maximum(c, 0.0)
        k = np.asarray(k, dtype=float)

        rates = np.zeros(len(self.reactions), dtype=float)
        for j, rxn in enumerate(self.reactions):
            rate = k[j]
            # reactants is a tuple of species indices, possibly repeated if you encode stoich >1
            for i in rxn.reactants:
                rate *= c[i]
            rates[j] = rate
        return rates


    def sample_noise(self, C):
        C_noisy = C + self.noise_level * self.rng.standard_normal(C.shape)
        return C_noisy

    @staticmethod
    def _ReLU(x_array: np.ndarray) -> np.ndarray:
        return np.maximum(0, x_array)



    @staticmethod
    def _add_extra_species(composition_matrix: np.ndarray) -> np.ndarray:
        """
        Generate extra species to complete the species space.
        Inputs:
            composition_matrix (np.ndarray): Composition matrix of shape (n_atoms, n_species)
        Returns:
            np.ndarray: Extended composition matrix of shape (n_atoms, n_total_species)
        """
        N_a, N_s = composition_matrix.shape
        existing_species = set("".join(map(str, composition_matrix[:, i])) for i in range(N_s))
        alphabet = [0, 1]
        possible_species = list(itertools.product(alphabet, repeat = N_a))
        to_add = []
        for species in possible_species :
            if "".join(map(str, species)) not in existing_species and "".join(map(str, species)) != "0" * N_a:
                to_add.append(np.array(species))
            
        extended_composition_matrix = composition_matrix.copy()
        extended_composition_matrix = np.concatenate([extended_composition_matrix, np.array(to_add).T], axis=1)

        return extended_composition_matrix  
    
    def _compute_sensitivites(self, c0, k):
        """
        Compute sensitivities of concentrations to parameters.
        Variables and parameters:
            c0 (list[float]): Initial concentrations of shape (n_species,)
            C(t) (np.ndarray): Species concentrations at time t (n_species,)
            U(t) (np.ndarray): Sensitivity matrix at time t (n_species, n_params) 
            r(t) (np.ndarray): Concentration normalized reaction rates at time t (n_reactions,)
            theta = phi + W (np.ndarray): Parameters (reaction rates and stoichiometries) (n_params,)
            f(t)  = dC(t)/dt (np.ndarray): Time derivative of concentrations at time t (n_species,)
            Gamma (np.ndarray): Composition matrix (n_atoms, n_species)
            B_null (np.ndarray): Basis for null space of composition matrix (n_species, n_null = (n_species - rank(Gamma)))
            W (numpy.ndarray): Weight matrix over null space giving nu = BW
                Obtained by solving nu = BW(n_null, n_reactions)
            nu (np.ndarray): Latent signed matrix (n_species, n_reactions)
            alpha = self._ReLU(nu)  #reaction orders: (n_species, n_reactions)
            beta = -nu               #stoichiometric matrix: (n_species, n_reactions)
            theta = np.concatenate([phi, W_ab])  #(n_reactions + n_null * n_reactions,)
        Key equations:
            dC/dt = f(t) = beta * r(t)
            dU/dt = A * U + B
            A = df(t)/dC(t) (n_species, n_species)
            B = df(t)/dtheta (n_species, n_params)
            
        Returns:
            C (np.ndarray): Concentrations over time of shape (n_timepoints, n_species)
            U (np.ndarray): Sensitivities over time of shape (n_timepoints, n_species, n_params)
        """
        Gamma_true = self.composition_matrix
        N_s = Gamma_true.shape[1] #number of true species
        N_a = Gamma_true.shape[0] #number of atoms
        N_ps = sum([math.comb(N_a, i) for i in range(1, N_a + 1)])
        Gamma = self._add_extra_species(Gamma_true)  #(N_a, N_ps)
        B_null = null_space(Gamma) # (N_ps, N_null)
        N_null = B_null.shape[1]
        nu_true = -self._build_stoichiometric_matrix()  #(N_s, N_reactions)
        nu = np.concatenate([nu_true, np.zeros((N_ps - N_s, nu_true.shape[1]))], axis=0)
        N_t = self.n_timepoints
        
        #solve for W in nu = B_null * W
        W, _, _, _ = lstsq(B_null, nu, rcond=None)  #(N_null, N_reactions)
        
        phi = np.array([np.log(k_i) for k_i in k]) #(N_reactions,)
        W_ab = W.flatten() #(N_null * N_reactions,)
        theta = np.concatenate([phi, W_ab])  #(N_reactions + N_null * N_reactions,)
        N_p  = len(theta) # Number of parameters
        
        c0 = np.append(c0, self.eps * np.ones(N_ps - N_s))
        c0 = np.array(c0, dtype=float)
        U0 = np.zeros((N_ps, N_p))
        
        
        alpha = self._ReLU(nu)  #reaction orders: (N_ps, N_reactions)
        beta = -nu               #stoichiometric matrix: (N_ps, N_reactions)
  
                
                
        def A(k, c, alpha, beta):
            """
            alpha: reaction orders (num_species, num_reactions)
            beta: stoichiometric matrix (num_species, num_reactions)
            A: rows = species rates, cols = species concentrations
            df(t)/dC (num_species, num_species)
            dfi/dCj = sum_k beta_{i,k} * r_k * alpha_{j,k} */ C_j
            """
            c_floor = 1e-6 #larger floor to avoid numerical issues
            N_r = beta.shape[1]
            N_ps = beta.shape[0]
            A = np.zeros((N_ps, N_ps)) # (num_reaction, num_species)
            r = self._rate_vector(k, c, alpha, c_floor)  # (num_reactions,)
            for j in range(N_ps):
                row = []
                for i in range(N_ps):
                    a_ij = np.sum([beta[i, k] * r[k] * alpha[j, k] / max(c[j],c_floor) for k in range(N_r)])
                    A[i, j] = a_ij
            return A
                
        def B(k, c, W, nu, B_null, N_phi):
            """
            alpha: reaction orders (num_species, num_reactions)
            beta: stoichiometric matrix (num_species, num_reactions)
            B: rows = species rates, cols = parameters
            df_i(t)/dphi_j = beta(i,j) * r_j(t) (num_species, num_reactions)
            df(t)/dW_ab = sum_m [(dfi/dalpha(m,b)) (dalpha(m,b_/dW(a,b))
                            + (dfi/dbeta(m,b)) (dbeta(m,b)/dW(a,b))]
            """
            c_floor = 1e-6 #larger floor to avoid numerical issues
            beta = -nu
            alpha = self._ReLU(nu)
            N_p = N_phi + W.shape[0] * W.shape[1]
            N_r = beta.shape[1]
            N_ps = beta.shape[0]
            B = np.zeros((N_ps, N_p))  #(num_species, num_params)
            r = self._rate_vector(k, c, alpha, c_floor)  # (num_params,)
            for j in range(N_p):
                for i in range(N_ps):
                    if j < N_phi:
                        #dfi/dphi_j
                        B[i, j] = beta[i, j] * r[j]
                    
                    else:
                        #dfi/dalpha_mj
                        a = (j - N_phi) // N_r
                        b = (j - N_phi) % N_r
                        for m in range(N_ps):
                            dnu_mb_dW_ab = B_null[m, a]
                            dalpha_mb_dW_ab = dnu_mb_dW_ab if nu[m, b] > 0 else 0
                            dbeta_mb_dW_ab = -dnu_mb_dW_ab
                            #dfi/dbeta_mj
                            if i == m:
                                dfi_dbeta_mb = r[b]
                            else:
                                dfi_dbeta_mb = 0
                            #dfi/dalpha_mj
                            dfi_dalpha_mb = beta[i, b] * r[b] * np.log(max(c[m], c_floor))
                            #dfi/dW_ab
                            B[i, j] += dfi_dalpha_mb * dalpha_mb_dW_ab + dfi_dbeta_mb * dbeta_mb_dW_ab
                    
            return B
                

        
        def rhs(t, z, k, W, B_null):
            N_phi = len(k)
            nu = B_null @ W
            alpha = self._ReLU(nu)
            N_ps = nu.shape[0]
            N_p = N_phi + W.shape[0] * W.shape[1]
            N_t = self.n_timepoints
            beta = -nu
            
            c = z[:N_ps]
            c = np.maximum(c, 0.0)


            U_flat = z[N_ps:]
            
            
            U = U_flat.reshape((N_ps, N_p))
            A_c = A(k, c, alpha, beta)
            B_c = B(k, c, W, nu, B_null, N_phi)
            
            dc_dt = beta @ self._rate_vector(k, c, alpha, c_floor=1e-6)
            dU_dt = (A_c @ U + B_c).flatten()
            
            dz_dt = np.concatenate([dc_dt, dU_dt])
            return dz_dt
        
        z0 = np.concatenate([c0, U0.flatten()])
        S = solve_ivp(rhs, (0, self.t_end), z0, t_eval=np.linspace(0, self.t_end, self.n_timepoints), args=(k, W, B_null), method="BDF", rtol=1e-9, atol=1e-12)
        if not S.success:
            print("message:", S.message)
        C = S.y[:N_ps, :].T
        U = S.y[N_ps:, :].T.reshape((N_t, N_ps, N_p))
        return C, U
                      
    def _compute_fim_output(self, c0, k, summed = True):
        """
        Compute Fisher Information Matrix.
        """
        C, U = self._compute_sensitivites(c0, k)
        out_idx = self.partition.output_indices[0]
        sigma = self.noise_level
        N_p = U.shape[-1] #number of parameters
        fim = np.zeros((self.n_timepoints, N_p, N_p))
        for ti in range(self.n_timepoints):
            Ut = U[ti, out_idx]
            fim[ti] += (1 / (sigma ** 2)) * np.outer(Ut, Ut)
        if summed:
            fim = np.sum(fim, axis=0)
        return C, fim
     
     
    @staticmethod
    def _SVD(fim):
        U, s, Vh = svd(fim)
        return U, s, Vh
     
    

    def simulate_mean(self, c0, k):
        """
        Mean simulation for one sample (no FIM).
        c0: (N_s,)
        k:  (N_r,)
        """
        S = self._build_stoichiometric_matrix()
        c0 = np.asarray(c0, dtype=float)
        t_eval = np.linspace(0, self.t_end, self.n_timepoints)

        def rhs(t, c):
            r = self._rate_vector_true(c, k)
            return S @ r

        sol = solve_ivp(
            rhs,
            (0, self.t_end),
            c0,
            t_eval=t_eval,
            method="BDF",   # stable for stiff-ish kinetics
            rtol=1e-9,
            atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        return t_eval, sol.y.T


    
    def setup_hardcoded_single_step(self):
        """
        Hardcode a 1-step cascade with intermediate:
            S_1 + S_2 → S_out
        With a valid atom-conserving composition matrix.
        """
        # Partition matches naming convention
        partition = Partition(inputs=['S_1', 'S_2', 'S_3'], hidden=[], output='S_out')
        #S_3 is not really hidden but helps stabilize the composition matrix
        # Index order: [S_1, S_2, S_3, S_out]
        # Atoms: A, B, and C
        composition_matrix = np.array([
            [1, 0, 1, 0],  # Atom A
            [0, 1, 0, 1],  # Atom B
            [0, 1, 1, 0],  # Atom C
        ], dtype = int)


        # Use indices [0,1,2] for species in partition.species
        reactions = [
            Reaction(reactants=(0, 1), products=(2, 3)),  # S_1 + S_2 → S_out
        ]

        return composition_matrix, partition, reactions


    
    def setup_hardcoded_1latent_example(self):
        """
        Hardcode a 1-step cascade with intermediate:
            S_1 + S_2 → S_3
            S_3 → S_4 + S_out
        With a valid atom-conserving composition matrix.
        """
        # Partition matches naming convention
        partition = Partition(inputs=['S_1', 'S_2'], hidden=['S_3', 'S_4'], output='S_out')
        # Index order: [S_1, S_2, S_3, S_4, S_out]
        # Atoms: A, B, and C
        composition_matrix = np.array([
            [1, 0, 1, 1, 0],  # Atom A
            [0, 1, 1, 0, 1],  # Atom B
            [0, 1, 1, 1, 0],  # Atom C
        ], dtype = int)


        # Use indices [0,1,2,3,4] for species in partition.species
        reactions = [
            Reaction(reactants=(0, 1), products=(2,)),  # S_1 + S_2 → S_3
            Reaction(reactants=(2,), products=(3, 4)),   # S_3 → S_4 + S_out
        ]

        return composition_matrix, partition, reactions


    def _fim_singular_values(self, fim: np.ndarray) -> np.ndarray:
        # since fim is symmetric psd (should be), singular values == eigenvalues (>=0)
        return np.linalg.svd(fim, compute_uv=False)

    def _E_from_fim(self, fim: np.ndarray) -> float:
        """
        E-optimality = smallest *numerically active* singular value of FIM.
        (For symmetric PSD FIM, svd singular values == eigenvalues.)
        """
        s = np.linalg.svd(fim, compute_uv=False)
        if s.size == 0:
            return 0.0

        return float(s[-1])

    def _E_from_fim_restricted(self, fim: np.ndarray, idx_keep: np.ndarray) -> float:
        fim_r = fim[np.ix_(idx_keep, idx_keep)]
        return self._E_from_fim(fim_r)

    def _relevant_param_indices(self, rtol: float = 1e-10, atol: float = 1e-14) -> np.ndarray:
        N_s = len(self.partition.species)
        N_r = len(self.reactions)
        B_null = self._get_B_null()
        N_null = B_null.shape[1]

        keep = [True] * N_r  # keep all phi

        for a in range(N_null):
            col = B_null[:N_s, a]
            # relative threshold against the column scale
            scale = np.max(np.abs(B_null[:, a]))
            thresh = max(atol, rtol * scale)
            touches_true = np.max(np.abs(col)) > thresh
            keep.extend([touches_true] * N_r)

        return np.flatnonzero(np.array(keep, dtype=bool))

    def _get_B_null(self) -> np.ndarray:
        """B_null depends only on the composition matrix (mechanism), not on c0/k."""
        return self._compute_B_null_full_and_true()[0]
    
    def _compute_B_null_full_and_true(self):
        Gamma_true = self.composition_matrix
        N_s = Gamma_true.shape[1]
        N_a = Gamma_true.shape[0]
        N_ps = sum([math.comb(N_a, i) for i in range(1, N_a + 1)])

        Gamma = self._add_extra_species(Gamma_true)   # (N_a, N_ps)
        B_null_full = null_space(Gamma)               # (N_ps, N_null)
        B_null_true = null_space(Gamma_true)          # (N_s, N_null)
        return B_null_true, B_null_full
    
    def grid_search(self, sampling_dim, c_ranges, k_ranges, parallel=False):
        samples = []

        # precompute restricted param indices ONCE per mechanism
        idx_keep = self._relevant_param_indices()

        c0_grid = [
            np.linspace(c_ranges[s][0], c_ranges[s][1], sampling_dim)
            if c_ranges[s][0] != c_ranges[s][1] else [c_ranges[s][0]]
            for s in range(len(self.partition.species))
        ]
        k_grid = [
            np.linspace(k_ranges[r][0], k_ranges[r][1], sampling_dim)
            if k_ranges[r][0] != k_ranges[r][1] else [k_ranges[r][0]]
            for r in range(len(self.reactions))
        ]

        for sample in itertools.product(*c0_grid, *k_grid):
            c0 = sample[:len(self.partition.species)]
            k  = sample[len(self.partition.species):]

            C, fim = self._compute_fim_output(c0, k, summed=True)

            E = self._E_from_fim(fim)
            E_restricted = self._E_from_fim_restricted(fim, idx_keep)

            samples.append({
                "c0": c0,
                "k": k,
                "FIM": fim,
                "FIM_r": fim[np.ix_(idx_keep, idx_keep)],
                "singular_values": self._fim_singular_values(fim),
                "singular_values_r": self._fim_singular_values(fim[np.ix_(idx_keep, idx_keep)]),
                "E": E,
                "E_restricted": E_restricted,
            })

        return samples

    
    def generate_batch(
        self,
        c0_list,
        k_list,
        compute_E: bool = False,
        compute_E_restricted: bool = False,
        return_fim: bool = False,
        return_label: bool = False,
    ):
        c0_arr = np.asarray(c0_list, dtype=float)
        k_arr  = np.asarray(k_list, dtype=float)

        B, N_s = c0_arr.shape
        assert N_s == len(self.partition.species)
        assert k_arr.shape == (B, len(self.reactions))

        t_eval = np.linspace(0, self.t_end, self.n_timepoints)
        out_idx = int(self.partition.output_indices[0])

        y_list, y_full_list = [], []
        E_list, E_r_list = [], []
        fim_list = []
        singular_values_list = []
        singular_values_r_list = []
        labels = []

        # If you want restricted-E, we need idx_keep; easiest is compute it on first FIM pass.
        idx_keep = None
        
        B_null_true, B_null_full = self._compute_B_null_full_and_true()

        for b in range(B):
            t, C = self.simulate_mean(c0_arr[b], k_arr[b])
            C_noisy = self.sample_noise(C)
            y_list.append(C_noisy[:, out_idx])
            y_full_list.append(C_noisy)

            if compute_E or compute_E_restricted or return_fim:
                C_fim, FIM = self._compute_fim_output(c0_arr[b], k_arr[b], summed=True)
                if return_fim:
                    fim_list.append(FIM)

                if compute_E:
                    E_list.append(self._E_from_fim(FIM))

                if compute_E_restricted:
                    if idx_keep is None:
                        # build idx_keep from B_null used inside sensitivities
                        # easiest: modify _compute_sensitivites to optionally return B_null (and N_s, N_r known)
                        # For now, assume you return B_null as third output:
                        # Ctmp, Utmp, B_null = self._compute_sensitivites(..., return_B_null=True)
                        pass

                    E_r_list.append(self._E_from_fim_restricted(FIM, idx_keep))

            if return_label:
                labels.append({
                    "mechanism": self.mechanism,
                    "k": k_arr[b].copy(),
                    "c0": c0_arr[b].copy(),
                    "reactions": self.reactions,
                })

        batch = {
            "t": torch.tensor(t_eval, dtype=torch.float32),
            "c0": torch.tensor(c0_arr, dtype=torch.float32),
            "k": torch.tensor(k_arr, dtype=torch.float32),
            "y": torch.tensor(np.stack(y_list, axis=0), dtype=torch.float32),
            "y_full": torch.tensor(np.stack(y_full_list, axis=0), dtype=torch.float32),

            "out_index": torch.tensor(out_idx, dtype=torch.int64),
            "B_null_true": torch.tensor(B_null_true, dtype=torch.float32),  # (N_s, N_null)
            "B_null_full": torch.tensor(B_null_full, dtype=torch.float32),  # (N_ps, N_null)
        }


        if compute_E:
            batch["E"] = torch.tensor(E_list, dtype=torch.float32)
        if compute_E_restricted:
            batch["E_restricted"] = torch.tensor(E_r_list, dtype=torch.float32)
        if return_fim:
            batch["FIM"] = fim_list
        if return_label:
            batch["label"] = labels

        return batch






    