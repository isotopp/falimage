Run a model on fal.ai, and grab the result.

0. https://docs.astral.sh/uv/getting-started/installation/ 
   Install uv.
1. `git clone https://github.com/isotopp/falimage`
2. `uv sync`
3. `cp sample-dotenv.txt .env`
4. `uv run pytest`
5. `uv run falimage --help`

and then

```
uv run falimage -m seedream -i landscape_4_3 -f tintin
```

This will load prompts/tintin.txt and run the request against fal.ai,
then store the result in assets/.
