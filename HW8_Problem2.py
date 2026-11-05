import numpy as np


np.random.seed(42)
N_SIM = 200000
T_WIN = 5.0

print("--- Problem 2 ---")

# 2(c)
lam_star = 3.0
lam_sky = 2.0
q = 0.4
rate_star = 1.2
rate_sky = 0.8

emit_star = np.random.poisson(lam_star * T_WIN, N_SIM)
emit_sky = np.random.poisson(lam_sky * T_WIN, N_SIM)
det_star_m1 = np.random.binomial(emit_star, q)
det_sky_m1 = np.random.binomial(emit_sky, q)
tot_m1 = det_star_m1 + det_sky_m1

det_star_m2 = np.random.poisson(rate_star * T_WIN, N_SIM)
det_sky_m2 = np.random.poisson(rate_sky * T_WIN, N_SIM)
tot_m2 = det_star_m2 + det_sky_m2

corr_m1 = np.corrcoef(det_star_m1, det_sky_m1)[0, 1]
corr_m2 = np.corrcoef(det_star_m2, det_sky_m2)[0, 1]

p_6_given_10_m1 = np.mean(det_star_m1[tot_m1 == 10] == 6)
p_6_given_10_m2 = np.mean(det_star_m2[tot_m2 == 10] == 6)

t_star = np.random.exponential(1.0 / rate_star, N_SIM)
t_sky = np.random.exponential(1.0 / rate_sky, N_SIM)

star_first = (t_star < t_sky)
t_first = np.minimum(t_star, t_sky)
M = np.maximum(t_star, t_sky)

p_star_first_hat = np.mean(star_first)
p_star_first_1s_hat = np.mean(star_first & (t_first <= 1.0))
e_M_hat = np.mean(M)
corr_time_type = np.corrcoef(t_first, star_first.astype(float))[0, 1]

print(f"2(c) Method 1: Star mean={np.mean(det_star_m1):.4f}, var={np.var(det_star_m1):.4f}, "
      f"Sky mean={np.mean(det_sky_m1):.4f}, var={np.var(det_sky_m1):.4f}, "
      f"Total mean={np.mean(tot_m1):.4f}, var={np.var(tot_m1):.4f}, "
      f"Corr={corr_m1:.4f}, P(6|10)={p_6_given_10_m1:.4f}")
print(f"2(c) Method 2: Star mean={np.mean(det_star_m2):.4f}, var={np.var(det_star_m2):.4f}, "
      f"Sky mean={np.mean(det_sky_m2):.4f}, var={np.var(det_sky_m2):.4f}, "
      f"Total mean={np.mean(tot_m2):.4f}, var={np.var(tot_m2):.4f}, "
      f"Corr={corr_m2:.4f}, P(6|10)={p_6_given_10_m2:.4f}")
print(f"2(c) Waits: P(star first)={p_star_first_hat:.4f}, P(star first & <=1s)={p_star_first_1s_hat:.4f}, "
      f"E[M]={e_M_hat:.4f}, Corr(T, type)={corr_time_type:.4f}\n")
