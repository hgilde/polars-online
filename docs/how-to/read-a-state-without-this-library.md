<!-- kind: how-to -->
# Read a state without this library

This how-to is for anyone who wants a program that does not use
polars-online, in Python or another language, to read what a model bank
has learned.

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads)'s
[state](../GLOSSARY.md#saving-and-serving) is everything it has learned.
`bank.save` writes it as a binary file, which only this library loads.
`bank.write_json` exports the same state as JSON, which any JSON reader
can parse.

## Prerequisites

These steps need the library, to export the state, and a JSON reader:

- polars-online installed, and
  [Example data](../EXAMPLE-DATA.md) built. The example reads
  `df`, a frame of 400 rows with the features `x0` and `x1` and the target
  `y`.
- A bank that has learned from rows, or a state file loaded with
  `po.ModelBank.load` ([Save a state](save-a-state.md)).
- A JSON reader in the program that reads the export. The example uses
  Python's `json` module.

## Export the state and read it

Export the state from Python, and read the JSON in the other program.

1. **Export the state with `bank.write_json()`,** which returns the text,
   or write it to a file with `bank.write_json(path)`. Every export is read
   back and checked against the state before you get it, and one that does
   not match raises `ValueError`.
2. **Keep the binary file too, for resuming,** since `po.ModelBank.load`
   reads only that form.
3. **Parse the JSON with any reader.**
4. **Turn the strings `"nan"`, `"inf"` and `"-inf"` back into numbers.**
   JSON has no literal for `NaN` or `±inf`, and an ordinary state can hold
   them, as a spec with `half_life=inf` (no forgetting) does. The strings
   are the spelling a spec's `half_life` takes.

This code exports a bank whose spec forgets nothing, and reads its
`half_life` back as a number. It uses `df` from [Example
data](../EXAMPLE-DATA.md):

```python
import json

bank = po.ModelBank([po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=float("inf"))])
bank.fit_predict(df)
text = bank.write_json()                  # 1. everything save() writes, as JSON
bank.write_json("bank.json")              # 1. the same, to a file
state = json.loads(text)                  # 3. any JSON reader
half_life = state["specs"][0]["half_life"]
print(f"{state['magic']}: half_life {half_life!r}, as a number {float(half_life)}")   # 4. the string, as a number
```

```text
polars-online-bank: half_life 'inf', as a number inf
```

## Check that it worked

**The export's `magic` field reads `polars-online-bank`,** as the example
prints, and each number your program reads matches the bank's. For the
coefficients, compare with `bank.coef()` ([Read the
coefficients](read-the-coefficients.md)).
