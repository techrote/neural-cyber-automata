---
title: Glossary
kind: reference
summary: Plain-language meanings of project terms and abbreviations.
owns: [glossary]
verified: 2026-10-13
---

# Glossary

No terms are defined in code yet; the "Where" column fills in as code lands.

| Term | Plain meaning | Where it's defined in code |
|---|---|---|
| NCA (neural cellular automaton) | A grid where every cell runs the same tiny neural network on itself and its neighbours each step. Behaviour emerges from many cells following one learned local rule. | not yet |
| Nanite | A cell of a learned "living" material in the game world. | not yet |
| Hidden state / state plane | A few private numbers each nanite carries between steps, like a short memory and a chemical signal. | not yet |
| Intent | What a nanite asks to do this step (move, attach, convert, signal). The world's rules decide whether it happens. | not yet |
| Signal field | A coarse layer the game writes over an area (alarm, attract, repair) that nanites can sense. | not yet |
| Perception filter | The fixed sums and differences of neighbouring cells a nanite looks at (similar to edge detection). | not yet |
| Alive masking | Cells with no living neighbour are reset to empty, so growth stays connected. | not yet |
| Sample pool | A stored set of earlier training states that training restarts from, so learned shapes stay stable over long runs. | not yet |
| Stochastic update | Each step only a random subset of cells updates, so no central clock is needed. | not yet |
| Evolution strategies (ES) | A training method: try many slightly different copies of the network, keep moving towards the better-scoring ones. Needs no gradients. | not yet |
| Quantisation | Converting weights to small integers (for example 8-bit) so inference is small, fast and identical on every platform. | not yet |
| Nanosand | The proposed simplified toy sand world used for training (D-007). | not yet |
| Lane | A device and software combination training can run on: CPU, Kepler K80 (legacy environment), modern NVIDIA GPU. | not yet |
| Behaviour suite | A fixed, versioned set of test scenarios and seeds used to score a trained behaviour. | not yet |
