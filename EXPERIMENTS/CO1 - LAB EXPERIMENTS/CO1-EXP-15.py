import matplotlib.pyplot as plt

prior = 0.30
p_word_spam = 0.70
p_word_not_spam = 0.10

posterior = (p_word_spam * prior) / (
    p_word_spam * prior +
    p_word_not_spam * (1 - prior)
)

print("Probability email is spam:", posterior)

plt.bar(["Prior Spam", "After 'prize'"],
        [prior, posterior])
plt.ylabel("Probability")
plt.show()
