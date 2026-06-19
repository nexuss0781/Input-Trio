# Input-Trio

Unified input layer for the Nexus Transformer: **Tokenize → Embed → Position-Encode**.

## Structure

```
Input-Trio/
├── Component-1.1_Tokenizer/           # Tokenizer (Python stub)
├── Component-1.2_Token-Embedding/     # HFAQE embedding (submodule)
├── Component-1.3_Positional-Encoding/ # HDPE positional encoding (submodule)
├── master-input/                      # Unified orchestrator (C++17)
│   ├── src/
│   │   ├── input_engine.cpp           # InputEngine class
│   │   ├── training.cpp               # training pipeline
│   │   └── main.cpp                   # demo
│   ├── tests/
│   │   └── test.cpp                   # integration suite
│   └── CMakeLists.txt
├── .gitmodules
└── README.md
```

## Submodules

```bash
git submodule update --init --recursive
```

- [Nexuss_Embedding](https://github.com/nexuss0781/Nexuss_Embedding) — Component-1.2 Token-Embedding (HFAQE)
- [Positional-Encoding](https://github.com/nexuss0781/Positional-Encoding) — Component-1.3 (HDPE)

## Build

```bash
cd master-input
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/master_main   # demo
./build/master_test   # integration tests
```
