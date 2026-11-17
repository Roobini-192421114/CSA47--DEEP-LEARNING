# Conditional probability using Bayes' rule
# Machine A makes 60% of products and has a 2% defect rate.
# Machine B makes 40% of products and has a 5% defect rate.

p_a = 0.60
p_b = 0.40
p_defect_given_a = 0.02
p_defect_given_b = 0.05

p_defect = p_defect_given_a * p_a + p_defect_given_b * p_b
p_a_given_defect = (p_defect_given_a * p_a) / p_defect

print("P(Defective):", round(p_defect, 4))
print("P(Machine A | Defective):", round(p_a_given_defect, 4))
print("Percentage:", round(p_a_given_defect * 100, 2), "%")
