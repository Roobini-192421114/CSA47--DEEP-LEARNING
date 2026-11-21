print("EXPERIMENT 64: RECURSIVE NEURAL NETWORK FOR SENTIMENT")
# A tiny recursive tree example: combine word-level sentiment scores bottom-up.
scores={"good":1.0,"excellent":1.0,"bad":-1.0,"poor":-1.0,"movie":0.0,"service":0.0}
def compose(node):
    if isinstance(node,str): return scores.get(node,0.0)
    left=compose(node[0]); right=compose(node[1])
    return (left+right)/2
trees={
    "good movie":("good","movie"),
    "poor service":("poor","service"),
    "excellent movie":("excellent","movie"),
    "bad service":("bad","service")
}
for sentence,tree in trees.items():
    value=compose(tree)
    print(f"{sentence:18s} -> {'Positive' if value>0 else 'Negative' if value<0 else 'Neutral'} ({value:.2f})")
print("The recursive function combines child-node representations to form a sentence representation.")
