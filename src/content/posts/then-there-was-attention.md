---
title: "AI Isn't Just Another Abstraction"
description: "LLM agents aren't just the next level of tooling—they're a qualitative leap. A look at the history of abstractions, what makes AI different, and what that means for us as engineers."
pubDate: 2026-11-24
tags: ["agents", "software-engineering"]
featured: false
readingTime: "7 min read"
---

I've heard a lot of software engineers say, "AI is just another tool, just another abstraction that we should use for our everyday work."

I think this is a very limited view of it, and we as humans don't properly comprehend the amount of intelligence the Frontier and Open models contain. I think a lot of us have intentionally ignored the pace of improvement and alignment in these models.

## Abstraction and Tools

Let's talk about abstraction first. In software engineering, we built tools to help us write code easily—letting an underlying system handle repeated tedious tasks so we can focus on the more important things.

* **Assembly → C:** Assembly was becoming too tedious to manage. The C compiler handled the mechanical work of converting C to machine instructions.
* **C → C++:** More abstractions, object-oriented patterns, and language-level features.
* **C → Python:** Wrapped C with memory management and a friendlier syntax—letting you think at a higher level.
* **C++ → JS engines:** V8 and friends brought the same idea to the browser.
* **JS → React:** Abstracted over DOM manipulation, optimized the render tree, and cut repetition so engineers could focus on building UI rather than hand-wiring updates.
* **Bare metal → Cloud / Kubernetes:** Abstracted hardware with virtualization and containerization, then gave us an orchestrator to manage it all.

Even the file system is an abstraction over a data structure.

As we can see, we have tools and abstractions of those tools all around us. We build those to be more efficient, fast, and productive—which is exactly what should be done. You cannot expect me to write a backend server in assembly for my business now. Earlier yes, now no!

## From LLMs to Agents

### LLMs

With the launch of the first RNNs with LSTM, and early work in NLP, we thought we had a shot at understanding language via computers. With Attention and Transformers, we filled a very big gap in language understanding: context and memory!

In the initial launch of GPT-2, it was a playground that could write poems, short paragraphs about things, sentences, basic Q&A, etc.

With GPT-3 onward, we got coding, better poems, essays, and advanced knowledge. It was like never before, and we did not give much thought to the fact that it would mog us and our knowledge this fast. We only thought of it as a helper that could understand our queries and give answers, not do things for us, which was the next step.

I think a lot of people also don't understand how and why LLMs are able to keep so much knowledge about language in just numbers, so let me help you understand it here.

At its core, an LLM is just billions of decimal numbers (parameters). A mid-sized open model has around **70 billion parameters**, and frontier models have **hundreds of billions to over a trillion**.

<details>
<summary><strong>🔢 How much knowledge fits in those numbers? (math breakdown)</strong></summary>

So how much actual information fits in those numbers?
* Raw English text has a lot of repetition (common words, grammar, formatting). All of English Wikipedia is around 25 GB of raw text, which compresses down to roughly **4 to 5 GB** of actual unique information.
* Research (*Physics of Language Models*) shows that neural networks can pack around **2 bits of pure, compressed factual knowledge per parameter**.
* For a 70B model: $70\text{ billion} \times 2\text{ bits} = 140\text{ billion bits} = 17.5\text{ GB}$ of already-compressed factual capacity. That is roughly **3x to 4x all of compressed English Wikipedia** inside a single mid-sized model—and frontier models are much larger.

And it doesn't just store facts like a giant zip file. Think about how you describe a person using just **2 numbers**: `[height, weight]`. You can only capture 2 traits. Now imagine representing every word and concept using **8,000 numbers** (dimensions) instead of 2. Even if each number could only be 0 or 1, just 100 numbers gives you $2^{100}$ unique combinations (more than the atoms in the universe). With thousands of continuous numbers passing through dozens of layers, the model doesn't memorize every sentence one by one. It learns reusable patterns and rules—like learning the relationship for "capital of" or "how a loop works" once, and applying it everywhere. That's how plain numbers hold both massive knowledge and reasoning.

</details>

Then came CoT (Chain of Thought). The model would reason and think inside itself before producing the actual output. Usually, this reasoning happens in **token space** itself, allowing the model to correct its trajectory if it's going in the wrong direction. It can also happen in **latent space**—where instead of forcing every thought to become a single English word token (which is a bottleneck), the model feeds its raw internal number vectors directly into the next step, letting it think across multiple paths at once before converting to text.

### Harness

As the models became more advanced with better instruction following, tool call invocations, and higher context windows, we got LLMs in our IDEs to help us on a tiny sidebar. It could take the file as input, answer my queries, and suggest inline changes.

Then came agent mode for the first time. I was hesitant to use it—how could I let this LLM touch the artistic code I write in my taste and elegantly? I gave it a chance anyways, and oh my, it wrote it better than me, that too in the file editor itself. I just had to click Accept for the code snippet and it was there. I could review it easily as it was just a single small file back then.

Soon the harnesses were becoming much more capable. A lot of different ones came: Cursor, Copilot with improvements, and Claude Code. This was the beginning of when we could have an LLM use its capabilities to the fullest for coding. We could choose to give it access to our repo, and it would read it, understand it, work on the assigned task, run bash commands, and use other tools provided to it.

In between came a lot of new techniques for making this system more autonomous - LLM councils, graphs of agents, hierarchies of agents, harness engineering, better harnesses. Then came OpenCode, even more autonomous agents like OpenClaw, and multiple products from the Frontier Labs for this.

## Why This Is NOT Just Another Abstraction

### 1. The Jump as an Abstraction and Tool

LLM agents are not a small, natural jump in abstraction or just another tool for us; it is a huge leap by many margins. The very idea that you could describe something in plain language and get it working as software was a goal, yes, but the pace is unimaginable. Everyone is racing to get the next jump first very fast, and with agents working continuously with humans to improve these systems themselves—which we call RSI (Recursive Self-Improvement)—it is giving an exponential speedup to all of this.

To comprehend how big of a leap this really is, look at the timeline:
* **Assembly to C:** ~20 years (1950s–1970s)
* **C to Python / JS:** ~20 years (1990s)
* **Bare metal to Cloud / Kubernetes:** ~15–20 years (2010s)
* Across all 60+ years of those abstractions, a human still wrote every single line of logic.

Now compare that to AI. In roughly **6 to 7 years**, we went from GPT-2 writing short paragraphs (2019), to inline autocomplete (2021), to sidebar chat (2023), to autonomous agents reading entire repos, running tests, and shipping features (2025–2026).

Going from C to Python was like moving from a hand tool to a power tool—you still hold it and make every cut yourself. Moving to AI agents is like going from a power tool to a team that reads your blueprint, builds the structure, and checks its own work.

### 2. The Jump in Our Role (From Intern-Buddy to Architect)

I have even heard a lot of people saying that this is just one level up of abstraction—in the sense that earlier there were a set of interns who were given tickets and their mentor/buddy decided what to give to whom and how to coordinate, and now you're that buddy. I think that's right to some extent, but not truly. This is not just one level up, it's many levels up. You're not even really at a software engineer level anymore, but more at an **architect level**. You could be maintaining multiple orchestrator agents who are managing sub-agents. You define workflows, states, and their transitions for how things should move at a high level, and review their work.

But you also have to be very careful. Even the best of the models are bound to make mistakes because of their auto-regressive nature. Yes, with tools and CoT to correct the model's trajectory, the sampling error reduces by a lot, but it's still there.

<details>
<summary><strong>📊 Why errors compound: the numbers (math breakdown)</strong></summary>

Here is why, in simple numbers:
* **No feedback loop:** Suppose a model is **95% accurate** per step (5% chance of error). Over a 20-step task done blind, one mistake can derail everything after it. Your chance of a clean run is $0.95^{20}$, which is only **~36%**. (Even at 99% accuracy over 100 steps, $0.99^{100}$ is still only **~37%**.)
* **With CoT + tools (compilers, tests, linters):** Instead of running 20 steps blind, you add checkpoints—say every 5 steps. Even if a 5-step chunk fails (~23% chance), the test catches it immediately before it ruins the rest. If the agent has a 70% chance of fixing a caught error on retry, that chunk's failure rate drops from **23% to ~7%** on the first retry, and **~2%** on the second.
* **Why it never hits 0%:** Two reasons. First, errors are often **correlated**—if the model misunderstands your requirement itself, retrying 5 times won't fix it. Second, your verifiers and tests won't catch 100% of edge cases.

</details>

Your agents can write code better than you, but when it breaks, you must know what broke and why. And to be even safer, you must guard your code very carefully using verifiers, evals, formatters, linters, tests, and guards to protect against these problems.

## Learnings

With this also came laziness. You see, software engineering is partly a craft—writing good software requires taste, patterns, and trade-offs, a lot of which was being delegated to an LLM for thinking and reasoning while I would just sit, prompt, and wait. In a way it was like a brainstorming buddy for me, but in a lot of cases we delegated our thinking to it.

I was talking with my friends about this, and with each one of them the conclusion was the same: with open models catching up and Frontier labs reducing the cost of inference over time, intelligence is going to be widely available, which used to be limited. Information was already widely available with all the search engines, blogs, posts, openly posted media, and texts, and now turning that into intelligence is getting commoditized too. Intelligence will no longer be the moat, the deciding factor, the important thing. I think we will slowly move to human creative taste, imagination, ideas, understanding trade-offs, and making decisions. In software engineering especially, if you can imagine and think of it, then you can probably build it.

The recent developments are the easiest proof of this—agents mogging the olympiads, solving 50-year-old problems, and even taking real shots at Millennium Prize problems.

With the advent of LLMs and agents, we have become quite fast, and with this pace comes the question of what about us, what will the role of humans be in this, and what happens to intelligence. I've tried to answer that here—intelligence is getting cheap, and the real work is moving to taste, judgment, and what we decide to build.
