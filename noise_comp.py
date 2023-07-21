import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def calculate_SNR(A_s, sigma_n):

    # Calculate the power of the signal
    P_s = (A_s)**2 / 2
    
    # Calculate the power of the noise
    P_n = sigma_n**2
    
    # Calculate the SNR in linear scale
    snr = P_s / P_n
    
    # Convert the SNR to dB scale
    # snr_dB = 10 * np.log10(snr)
    
    return snr


def compute_beta(final_noise_var, num_levels):

    if num_levels is None:
        return 0

    beta_init = 0.1

    def func(beta, var, x):
        return var - beta*(1 - (1-beta)**x)
    
    beta = fsolve(func, beta_init, args=(final_noise_var, num_levels))

    return beta[0]

if __name__ == "__main__":

    FINAL_VAR = 0.02
    # FINAL_VAR = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]
    NUM_NOISE_LEVELS = 2
    beta = compute_beta(FINAL_VAR, NUM_NOISE_LEVELS)
    print(f"Beta:{beta}")
    # x = np.arange(1, 10)
    # var = beta - beta*(1-beta)**x
    # std = np.sqrt(var)
    # y_fact = (1-beta)**(x/2)
    # # snr = calculate_SNR(y_fact, std)

    # plt.figure(figsize=(10,6))
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xlabel("# layers", fontsize=20, labelpad=20)

    # plt.plot(x, var, label="Noise VAR", linewidth=2)
    # plt.plot(x, std, label="Noise STD", linewidth=2)
    # plt.plot(x, y_fact, label="OG signal multiplier", linewidth=2)
    # # plt.plot(x, snr, label="SNR (dB)", linewidth=2)

    # plt.legend(loc="upper right", fontsize=16)

    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(f"images/var_{FINAL_VAR}_num_levels_{NUM_LEVELS}.png", dpi=300)
    # plt.close()