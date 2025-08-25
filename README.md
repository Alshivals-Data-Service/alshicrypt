# alshicrypt

Learned character–permutation cipher in PyTorch — train a pair of tiny models
(encoder/decoder) that memorize a random substitution key over an ASCII+punctuation
vocabulary. Designed for a **developer-friendly API**:

```python
import alshicrypt

crypt = alshicrypt.generate(epochs=200)  # train a new random key
enc = crypt.encode("Hello, World!")
dec = crypt.decode(enc)
assert dec == "Hello, World!"
````

> ⚠️ **Security disclaimer**
>
> This project is an educational/experimental demo. It is **not a secure cryptosystem** and should not be used for real security needs.
> If an attacker obtains the **model weights**, they can recover (or emulate) the character mapping in $O(|V|)$ queries by running the model on every vocabulary index—especially if the vocabulary/order is known.
> The strongest mitigation is to **never ship the decoder**, consider not shipping the encoder either, and keep both models **behind an authenticated API**. For real applications, prefer **standard, audited cryptography** (e.g., AES-GCM or ChaCha20-Poly1305) with proper key management (KMS/HSM), rotation, and access controls.

### Future work (research directions)

* Replace the random substitution with **standard authenticated encryption**; use ML only for auxiliary tasks (e.g., usability, format handling), not secrecy.
* Add **integrity and authenticity** guarantees (AEAD-like behavior) rather than confidentiality alone.
* Study secure deployment options (e.g., **keep models server-side**, trusted execution, or other isolation mechanisms).
* Provide formal security analysis and clear threat models; evaluate leakage via **model inversion/extraction** attacks.

---

## Table of contents

* [Research paper](#research-paper)
* [Install](#install)
* [Quick start](#quick-start)
* [Loading previously trained models](#loading-previously-trained-models)
* [Artifacts layout](#artifacts-layout)
* [Reproducibility and randomness](#reproducibility-and-randomness)
* [API reference](#api-reference)
* [Security notes & hardening](#security-notes--hardening)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

## Research paper

The technical write-up for this project is in **[`paper/README.pdf`](paper/README.pdf)** (generated from `paper/README.tex`). It documents the motivation, math, and code mapping for each class.

**Brief summary**

* **Vocabulary**: `Characters` concatenates ASCII letters, digits, punctuation, and whitespace into a fixed set $V$.
* **Key (cipher)**: `Cipher` samples a **random permutation** $\pi$ over $V$ and forms supervised pairs for both directions.
* **Model**: `Architecture` is a compact per-character classifier (Embedding $\rightarrow$ Linear) that outputs logits over $V$.
* **Objective**: Categorical cross-entropy; the task is multi-class classification over characters.
* **Training**: Two models are trained—an encoder (plaintext→ciphertext) and a decoder (ciphertext→plaintext)—until near-perfect accuracy on $V$.
* **Result**: The models **memorize** the permutation and invert each other exactly on the known vocabulary. This is a research demo, not a secure cryptosystem.

**Build the PDF**

```bash
cd paper
pdflatex -shell-escape README.tex
# If you use VS Code + LaTeX Workshop, make sure your recipe passes -shell-escape.
```

---

## Install

Requires **Python 3.9+** and **PyTorch 2.0+**.

Install directly from GitHub:

```bash
pip install "git+https://github.com/Alshivals-Data-Service/alshicrypt.git"
```

If you don’t have PyTorch yet, install a wheel appropriate for your system first:
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## Quick start

Train a new encoder/decoder and use them right away:

```python
import alshicrypt

# Train & save to a stable folder (optional but recommended)
crypt1 = alshicrypt.generate(epochs=200, outdir="artifacts/crypt-hello")

msg = "Hello, World!"
enc = crypt1.encode(msg)
dec = crypt1.decode(enc)

print("crypt1")
print("============================")
print(f"Original: {msg}")
print(f"Encoded:  {enc}")
print(f"Decoded:  {dec}")
assert dec == msg
```

* Out-of-vocabulary characters (e.g., emoji) are **passed through unchanged**.
* Each call to `generate()` produces a **new random key** unless you set a seed (see below).

---

## Loading previously trained models

You can reload a saved model pair later (or from a different process):

```python
import alshicrypt

# Train once (or do this in a separate script/process)
crypt1 = alshicrypt.generate(epochs=200, outdir="artifacts/crypt-hello")

# Load it later
crypt2 = alshicrypt.load("artifacts/crypt-hello")

msg = "Hello, World!"
enc2 = crypt2.encode(msg)
dec2 = crypt2.decode(enc2)

print("crypt2")
print("============================")
print(f"Original: {msg}")
print(f"Encoded:  {enc2}")
print(f"Decoded:  {dec2}")
assert dec2 == msg
```

Tip: when you call `generate()` without specifying `outdir`, it creates a timestamped
folder under `artifacts/`. You can always reuse that exact path via `crypt.outdir`.

---

## Artifacts layout

Each training run saves a small bundle (defaults to `artifacts/<timestamp>/`):

```
artifacts/<run-id>/
├─ encoder.pth              # final encoder weights
├─ decoder.pth              # final decoder weights
├─ encoder_best.pth         # best-by-accuracy snapshot (optional)
├─ decoder_best.pth         # best-by-accuracy snapshot (optional)
├─ vocab.json               # character set used
├─ mapping.json             # actual permutation + perm_seed (if any)
├─ config.json              # TrainerConfig used for this run
├─ history.json             # per-epoch metrics (loss/accuracy)
└─ README.txt               # short summary
```

To load a run: `alshicrypt.load("artifacts/<run-id>")`.

---

## Reproducibility and randomness

* **Key randomness (the permutation)**: controlled by `perm_seed`.

  * Default `perm_seed=None` → a **new random key** each time.
  * Set `perm_seed=123` to reproduce the same key later.

* **Training randomness**: controlled by `seed`.

  * Default `seed=None` → non-deterministic training (fine for this task).
  * Set `seed=123` if you need deterministic training behavior.

Example:

```python
# Reproducible key; random training
crypt = alshicrypt.generate(epochs=200, perm_seed=123, seed=None, outdir="artifacts/perm-123")

# Fully reproducible (key + training)
crypt = alshicrypt.generate(epochs=200, perm_seed=123, seed=123, outdir="artifacts/perm-123-train-123")
```

---

## API reference

### Top-level functions

```python
alshicrypt.generate(
    *, epochs=500, batch_size=32, lr=2e-3, emb_dim=64,
    seed=None, perm_seed=None, target_acc=1.0,
    outdir=None, use_cpu=False
) -> Crypt
```

Train a fresh encoder/decoder and return a **`Crypt`** runtime.
If `outdir` is not set, a timestamped folder is created under `artifacts/`.

```python
alshicrypt.load(path: str | pathlib.Path) -> Crypt
```

Load a `Crypt` runtime from an artifacts folder containing `encoder.pth`, `decoder.pth`, and `vocab.json`.

### `Crypt` runtime

```python
class Crypt:
    def encode(self, text: str) -> str: ...
    def decode(self, text: str) -> str: ...
    outdir: pathlib.Path | None  # where artifacts were loaded from (if available)
```

* `encode()` applies the learned encoder (plaintext → ciphertext).
* `decode()` applies the learned decoder (ciphertext → plaintext).
* Characters not in the saved vocabulary are passed through unchanged.

---

## Security notes & hardening

* **Do not ship the decoder** to untrusted environments. Keep it server-side when possible.
* You can remove metadata that leaks information about training or the key:

  * Safe to delete: `mapping.json`, `history.json`, `config.json`, and any `*_best.pth` snapshots.
  * Keep `vocab.json` unless you modify the loader to embed or infer the vocabulary.
* Use filesystem permissions for artifacts (e.g., `chmod 700` on directories and `chmod 600` on files).
* Remember: with access to model weights, an attacker can still query all characters and reconstruct the mapping; this demo is **not** a replacement for real cryptography. 

---

## Troubleshooting

* **`FileNotFoundError: Missing artifact ...`**
  Ensure you pass the **actual** run directory that contains `vocab.json`, `encoder.pth`, and `decoder.pth`.
  If you just trained, you can reuse `crypt.outdir` directly.

* **Different ciphertexts for the same message**
  This is expected if you trained two different crypts (different random keys).
  Use `perm_seed` to reproduce a specific key.

* **Slow training**
  This is a tiny model. On CPU it should converge quickly; on GPU it’s very fast.
  You can increase `batch_size` or `emb_dim` if desired.

---

## License

MIT © Samuel Cavazos
