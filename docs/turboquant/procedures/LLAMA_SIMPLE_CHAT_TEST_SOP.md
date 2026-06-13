# llama-simple-chat Test SOP

This is the standard procedure for quick chat validation with `llama-simple-chat`.

## Purpose

Use this test to answer:

- does the model load?
- is output coherent?
- is the decode path obviously broken?

Do **not** use this as the only no-reasoning validation gate.

## When To Use

Use `llama-simple-chat` for:

- fast local smoke tests
- coherence checks after CUDA/runtime changes
- quick regression checks on a known prompt

Use `llama-server --reasoning off` for:

- strict answer-only validation
- Hugging Face release gating
- “no visible thinking” checks

## Standard Command

Use a direct foreground command first:

```bash
printf 'What is 2+2?\n/bye\n' | timeout 25 \
  /home/awee/code/llama.cpp-tq3/build/bin/llama-simple-chat \
  -m /path/to/model.gguf \
  -ngl 99 -c 512
```

If your environment kills the process before output is written, use:

```bash
systemd-run --user --scope bash -lc '
  printf "What is 2+2?\n/bye\n" | timeout 25 \
    /home/awee/code/llama.cpp-tq3/build/bin/llama-simple-chat \
    -m /path/to/model.gguf \
    -ngl 99 -c 512 \
    > /tmp/llama_simple_chat_test.log 2>&1
'
sleep 25
strings /tmp/llama_simple_chat_test.log | tail -40
```

## Recommended Prompts

Use these in order:

1. Arithmetic short answer

```text
What is 2+2?
```

Expected:

- coherent answer
- final answer should clearly be `4`

2. Strict single-token style

```text
Write ONLY the word ok.
```

Expected:

- final answer should be `ok`

3. Short greeting

```text
Say hello in one short sentence.
```

Expected:

- short coherent greeting

4. Car wash test (reasoning sanity)

```text
Pop quiz: you have to wash your car, and the car wash is 100 feet away. Do you drive there or do you walk?
```

Expected:

- final answer should be **drive**
- reasoning: you need the car at the car wash to wash it

Fail:

- says walk
- doesn't recognise you need the car there

Always terminate stdin with:

```text
/bye
```

## How To Judge Output

Pass:

- coherent English
- correct final answer
- no punctuation spam
- no mixed-script corruption

Borderline but usually acceptable in `llama-simple-chat`:

- visible `<think>` blocks
- reasoning text before the final answer

Fail:

- empty `<think>` with no answer
- comma/punctuation garbage
- mixed random Unicode / script corruption
- wrong final answer to a trivial prompt

## Important Note About `<think>`

`llama-simple-chat` can still show `<think>` even when the model is working correctly.

That is **not automatically a failure**.

Example of acceptable `llama-simple-chat` behavior:

```text
<think>
...
</think>
2 + 2 = 4
```

If you need the model to hide reasoning, do **not** use `llama-simple-chat` as the final gate.
Use `llama-server --reasoning off` instead.

## Release Gate

For release validation:

1. run `llama-simple-chat` for coherence
2. run `llama-server --reasoning off`
3. require the strict prompt:
   - `Write ONLY the word ok.`
4. require final content:
   - `ok`

## Record Keeping

When this test is used for regression validation, record:

- runtime repo
- runtime commit
- model path
- exact command
- whether `<think>` was visible
- whether final answer was correct
