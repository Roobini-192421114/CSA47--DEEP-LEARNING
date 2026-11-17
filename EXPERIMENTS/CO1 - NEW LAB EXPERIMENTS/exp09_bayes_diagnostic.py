# Bayes' theorem for a medical diagnostic test
prevalence = 0.01       # 1% of population has the disease
sensitivity = 0.95     # P(positive | disease)
specificity = 0.90     # P(negative | no disease)
false_positive_rate = 1 - specificity

p_positive = sensitivity * prevalence + false_positive_rate * (1 - prevalence)
p_disease_given_positive = sensitivity * prevalence / p_positive

print("Disease prevalence:", prevalence)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Probability of positive test:", round(p_positive, 4))
print("P(Disease | Positive):", round(p_disease_given_positive, 4))
print("Percentage:", round(p_disease_given_positive * 100, 2), "%")
