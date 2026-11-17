import numpy as np

print("Floating-point precision demonstration")
a64 = np.float64(0.1)
b64 = np.float64(0.2)
a32 = np.float32(0.1)
b32 = np.float32(0.2)

print("\nfloat64: 0.1 + 0.2 =", repr(a64 + b64))
print("float32: 0.1 + 0.2 =", repr(a32 + b32))
print("float64 equals 0.3 exactly?", (a64 + b64) == np.float64(0.3))
print("float32 equals 0.3 exactly?", (a32 + b32) == np.float32(0.3))

large = np.float64(1e16)
print("\nLarge number + small number (float64):", large + 1 - large)
print("Small number may be lost due to finite precision.")

print("\nfloat32 machine epsilon:", np.finfo(np.float32).eps)
print("float64 machine epsilon:", np.finfo(np.float64).eps)
print("float64 normally provides more precision but uses more memory.")
