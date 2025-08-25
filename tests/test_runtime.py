import alshicrypt

def test_generate_and_roundtrip():
    crypt = alshicrypt.generate(epochs=50, target_acc=1.0, use_cpu=True)  # tiny run
    s = "Hello, World! 123"
    enc = crypt.encode(s)
    dec = crypt.decode(enc)
    assert dec == s
