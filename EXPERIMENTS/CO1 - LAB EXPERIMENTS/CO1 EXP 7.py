temp = [22, 25, 28, 30, 32]
humidity = [80, 75, 68, 60, 55]

n = len(temp)

mean_t = sum(temp) / n
mean_h = sum(humidity) / n

cov_tt = sum((x - mean_t)**2 for x in temp) / (n - 1)
cov_hh = sum((y - mean_h)**2 for y in humidity) / (n - 1)
cov_th = sum((temp[i] - mean_t) * (humidity[i] - mean_h)
             for i in range(n)) / (n - 1)

print("Covariance Matrix:")
print("[", round(cov_tt, 2), round(cov_th, 2), "]")
print("[", round(cov_th, 2), round(cov_hh, 2), "]")

print("Temperature:", temp)
print("Humidity:", humidity)
