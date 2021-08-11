Cosmos SDK Gym implements an OpenAI Gym interface for testing the Cosmos SDK using RL-assisted RNG to improve its code coverage and bug discovery. The methodology should be easily extensible to other software libraries written in Go or other languages.

#### Requirements
https://golang.org/ \
https://github.com/openai/gym \
https://github.com/ray-project/ray (RLlib) \
https://github.com/cytsai/cosmos-sdk (Cosmos SDK with RL-assisted RNG)

#### Installation
`pip install -e .`

#### Run
See `examples/helios/` and `examples/bst/`.
