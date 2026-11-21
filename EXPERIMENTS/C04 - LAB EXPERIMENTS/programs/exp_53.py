from collections import defaultdict, Counter
print("EXPERIMENT 53: RNN-STYLE NEXT WORD PREDICTION")
sentences=["deep learning is useful","deep learning models learn","recurrent networks process sequences","deep learning is powerful"]
next_words=defaultdict(Counter)
for sentence in sentences:
    words=sentence.lower().split()
    for a,b in zip(words[:-1],words[1:]): next_words[a][b]+=1
for prompt in ["deep","learning","recurrent"]:
    if next_words[prompt]:
        print(f"After '{prompt}' -> '{next_words[prompt].most_common(1)[0][0]}'")
    else: print(f"After '{prompt}' -> no learned next word")
print("This is a lightweight sequence-baseline demonstration. A trained RNN can learn longer context.")
