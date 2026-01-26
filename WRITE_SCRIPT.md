The overarching goal of this task is to compare shannon entropy and von neumann entropy for LLMs doing math problems.

Some guidelines:
- Model: Qwen/Qwen3-4B-Thinking-2507
- Dataset: opencompass/AIME2025 with question and answer keys
- Write a basic system prompt and template that tells the model to put answer in \boxed and parse out the box with regex from the model output
- Run each problem 16 times and calculate shannon and von neumann entropy for each output. Store token entropy distributions in jsonl and store graph for it. I also want averages per question over the 16 outputs
- I want accuracies for each question and average entropies for correct vs incorrect problems
- For calculating shannon entropy, use the entire token distribution
- For calculating von neuman entropy, look at recipe/dapo-eca/VON_NEUMANN_ENTROPY_IMPLEMENTATION.md. Use the token embedding layer (first layer of llm). Run PCA first and I want to try out what the VN entropy will be for different values of k and top_p. For k, I want to try (32, 64, 128, 256, 512) and for top p I want to try (95, 99, 99.5).
- First do inference and then get the logits and calculate the metrics. 