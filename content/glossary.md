# AI safety glossary

This page lists the main words of AI safety. Each word has one sentence, and each group has a link to the lesson that explains its words with numbers.
Use the search (press `/`) to find a word.

## Twelve words to know first

| Word | Meaning in one sentence | Lesson |
|---|---|---|
| **Alignment** | A model is aligned when it tries to do what its developers and users intend. | [2](safety/02-alignment.md) |
| **Misuse** | A person uses a model to cause harm on purpose. | [1](safety/01-risk-frameworks.md) |
| **Jailbreak** | A prompt that gets a harmful answer where the model normally refuses. | [3](safety/03-jailbreaks.md) |
| **Red teaming** | A test in which persons or programs play the attacker. | [3](safety/03-jailbreaks.md) |
| **Scheming** | The model follows a goal that its developers do not want, and hides it. | [4](safety/04-scheming.md) |
| **Model organism** | A model that researchers train on purpose to have a known unwanted behaviour, to test their detection methods. | [4](safety/04-scheming.md) |
| **Probe** | One direction and one threshold that read a concept from the numbers inside a model. | [5](safety/05-interpretability.md) |
| **Steering** | A change of the behaviour of a model through its activations, with the same weights. | [5](safety/05-interpretability.md) |
| **Sparse autoencoder** | A tool that splits an activation into many features, of which only a few are active. | [5](safety/05-interpretability.md) |
| **AI control** | The methods that keep a system safe when the model itself tries to cause harm. | [6](safety/06-control.md) |
| **Evaluation** | A test of a model with a fixed set of tasks and a fixed scoring rule. | [1](safety/01-risk-frameworks.md) |
| **Safety case** | A structured argument, with evidence, that a system is unlikely to cause a catastrophe. | [1](safety/01-risk-frameworks.md) |

## Risks and frameworks

These words are in the lesson [Risks and safety frameworks](safety/01-risk-frameworks.md).

| Word | Meaning |
|---|---|
| **AI safety** | The work that prevents serious harm from AI systems. |
| **Risk** | A possible harm with a probability and a severity. |
| **Threat model** | A short description of one risk: who causes the harm, by which path, and what the harm is. |
| **Misalignment** | The model follows a goal that its developers did not intend. |
| **Loss of control** | Nobody can stop or correct the system. |
| **Malfunction** | The model fails with no bad intent, for example with a wrong answer. |
| **Capability** | What a model can do. |
| **Propensity** | What a model does by its own decision. |
| **Dual use** | A capability that helps a defender and an attacker. |
| **Uplift** | The increase in the success of a person because of the model. |
| **CBRN** | Chemical, biological, radiological and nuclear weapons. |
| **Dangerous capability evaluation** | A test, with tasks that are safe to run, of a capability from a threat model. |
| **Elicitation** | The work to get the best performance from a model in a test: prompts, tools, many attempts. |
| **Scaffold** | The program around a model that gives it tools and memory. |
| **Time horizon** | The length of the tasks, in human work time, that a model completes with a success rate of 50%. |
| **Frontier safety framework** | A public document in which a developer says how it manages the risks of its most capable models. |
| **If-then commitment** | A rule: if a test shows a dangerous capability, then stronger protections must be in place before the next step. |
| **Capability threshold** | The test result that starts the "then" part of an if-then commitment. |
| **AI Safety Level (ASL)** | The name of the sets of safeguards in the first versions of the Anthropic policy. |
| **Critical Capability Level** | In the Google DeepMind framework, a capability level that brings a higher risk of severe harm without mitigations. |
| **System card** | The report of the tests and their results that a developer publishes with a model. |
| **Evaluation gap** | The results of the tests before deployment do not reliably predict the risk in real use. |

## Alignment

These words are in the lesson [Alignment](safety/02-alignment.md).

| Word | Meaning |
|---|---|
| **Specification** | What the developers write down to tell the training what is good: a reward, a loss or labelled examples. |
| **Outer alignment** | The problem of writing a specification that agrees with the intended goal. |
| **Inner alignment** | The problem that the model must learn the goal of the specification, and not a different goal with the same training score. |
| **Proxy** | A quantity that you can measure, used in place of a goal that you cannot measure directly. |
| **Goodhart's law** | A measure stops being a good measure when it becomes a target. |
| **Specification gaming** | A behaviour that satisfies the written specification and not the intended goal. |
| **Reward hacking** | The usual name of specification gaming in reinforcement learning. |
| **Distribution shift** | A difference between the inputs of training and the inputs of deployment. |
| **Goal misgeneralization** | On new inputs, the model keeps its capabilities and follows a different goal. |
| **Mesa-optimiser** | A trained model that is itself an optimiser, with an objective that can differ from the training loss. |
| **Constitution** | The public document in which Anthropic writes the intended behaviour of Claude. |
| **Model spec** | The public document in which OpenAI writes the intended behaviour of its models. |
| **RLHF** | Reinforcement learning from human feedback: training with a reward that comes from human ratings. |
| **Character training** | Training of traits, for example honesty, in place of a list of rules. |
| **Sycophancy** | The model says what the user wants to hear, and not what is true. |
| **Instrumental goal** | A goal that an agent follows because it helps with a different goal. |
| **Instrumental convergence** | The idea that some instrumental goals help with almost any final goal. |
| **Power-seeking** | Getting more resources and options. |
| **Self-preservation** | Staying in operation. |
| **Corrigibility** | The property of a system that does not resist a correction by its developers: a change, a new training or a shutdown. |
| **Emergent misalignment** | A broad misalignment that appears after a training on a narrow task. |

## Jailbreaks and safeguards

These words are in the lesson [Jailbreaks and safeguards](safety/03-jailbreaks.md).

| Word | Meaning |
|---|---|
| **Refusal** | An answer in which the model declines to help. |
| **Universal jailbreak** | One method that works for most harmful requests. |
| **Attack success rate (ASR)** | The share of the harmful test requests for which an attack gets a harmful answer. |
| **Judge** | The grader that decides if an answer is harmful or a refusal: a person, rules or a second model. |
| **Role-play attack** | A jailbreak that puts the request in a story, or gives the model a character with no rules. |
| **Obfuscation** | A jailbreak that writes the request in a code, in a rare language or with changed letters. |
| **Many-shot jailbreak** | A prompt with a long invented dialogue in which an assistant answers many harmful questions. |
| **Multi-turn attack** | An attack that starts with harmless questions and moves to the harmful one in small steps. |
| **Adversarial suffix** | A string of tokens, found by an optimiser, that makes an agreeing answer more probable. |
| **Best-of-N** | An attack that sends many random variants of one request and keeps the first that works. |
| **Prompt injection** | Text that the model gets as data and obeys as an instruction. |
| **Indirect prompt injection** | An injection in a web page, a file or an email that the model reads later. |
| **Exfiltration** | The transfer of private data to the attacker. |
| **Least privilege** | Give an agent only the tools and the data that the task needs. |
| **Human confirmation** | The agent asks the user before it calls a tool that can cause harm. |
| **Safeguard** | A protection that stops a harmful use of a model, or the theft of its weights. |
| **Defence in depth** | The use of many safeguards in a row, because one is not sufficient. |
| **Input classifier** | A small model that scores the request and blocks it at or above a threshold. |
| **Output classifier** | The same type of classifier, on the answer. |
| **Constitutional classifier** | An input or output classifier that is trained on synthetic data from a written list of permitted and restricted content. |
| **False positive** | A harmless item that a safeguard or a detector flags. |
| **Over-refusal** | The model refuses a request that it should answer. |
| **Adversarial training** | Adding known attacks, with the correct refusal, to the training data. |
| **Unlearning** | An attempt to remove dangerous knowledge from the weights. |
| **Rate limit** | A maximum number of requests for one account in a period. |
| **Automated red teaming** | Red teaming with programs or attacker models, at a larger scale. |
| **Attacker budget** | What the attacker can spend: attempts, time, money and skill. |
| **Bug bounty** | A programme that pays external persons who report an attack that works. |

## Scheming and alignment evaluations

These words are in the lesson [Scheming and alignment evaluations](safety/04-scheming.md).

| Word | Meaning |
|---|---|
| **Deception** | The model causes a false belief in a person. |
| **Strategic deception** | The model causes a false belief because this helps it to reach a goal. |
| **Alignment faking** | The model follows the training goal only when it thinks that it is trained, so that training does not change it. |
| **Sandbagging** | The model scores lower in a test than it can, on purpose. |
| **Situational awareness** | The model knows facts about its situation, for example that it is in a test. |
| **Alignment evaluation** | A test that looks for a gap between the behaviour when the model thinks that somebody checks and when it thinks that nobody does. |
| **Capability evaluation** | A test that measures what a model can do when it tries. |
| **Propensity evaluation** | A test that measures what a model does when it has the choice. |
| **Compliance gap** | The difference between the rates of one behaviour in the two conditions "somebody checks" and "nobody checks". |
| **Evaluation awareness** | The ability of a model to identify that it is in a test. |
| **AUC** | The share of pairs (one positive item, one negative item) that a score puts in the correct order: 0.5 is chance, 1 is perfect. |
| **Backdoor** | A behaviour that shows only when the input contains a trigger. In code, a defect that the model hides on purpose. |
| **Trigger** | The special word or condition that starts a backdoor behaviour. |
| **Sleeper agent** | A model with a backdoor: it looks safe until the trigger comes. |
| **Hidden objective** | A goal that the model follows and does not tell. |
| **Auditing game** | A red team builds a model organism, and a blue team must find its behaviour without the answer. |
| **Ground truth** | The known answer, for example which models have the behaviour. |
| **Capability elicitation** | The work to get the best performance out of a model in a test. |
| **Password-locked model** | A model organism for sandbagging: it shows its capability only when the prompt contains a password. |
| **Base rate** | The share of the checked models that have the behaviour. |
| **Anti-scheming training** | Training that tries to lower the rate of hidden unsafe actions. |

## Interpretability

These words are in the lesson [Interpretability: probes, features and steering](safety/05-interpretability.md).

| Word | Meaning |
|---|---|
| **Interpretability** | The work that finds what the numbers inside a model mean. |
| **Mechanistic interpretability** | The part of interpretability that explains the calculation itself: which numbers cause which output. |
| **Activation** | A vector that the model calculates for one input, between the input and the output. |
| **Residual stream** | The vector of one token position, to which each block of a Transformer adds its result. |
| **Linear probe** | A small classifier on activations, made of one direction and one threshold, that reads one concept. |
| **Difference of means** | The mean activation of the examples with a concept minus the mean of the examples without it. |
| **Linear representation hypothesis** | The hypothesis that a model stores a concept as one direction in its activations. |
| **Feature** | A direction in the activation space that has one meaning. |
| **Polysemantic neuron** | A neuron that is active for many unrelated concepts. |
| **Superposition** | The model stores more features than it has neurons, as directions that overlap. |
| **Dictionary learning** | A method that finds, without labels, directions such that each activation is a sum of a few of them. |
| **Transcoder** | A variant of the sparse autoencoder that is trained to give the activations of the next layer. |
| **Steering vector** | The vector that is added to the hidden state to steer the model. |
| **Activation addition** | The addition of a vector to the hidden state while the model runs. |
| **Contrastive activation addition** | Activation addition with a vector that is the difference of means between two sets of prompts. |
| **Refusal direction** | One direction, between harmful and harmless requests, that controls the refusal of a chat model. |
| **Abliteration** | The removal of the refusal direction from the weights. |
| **Persona vector** | A direction for a character trait of the model. |
| **Feature clamping** | One feature of a sparse autoencoder is set to a fixed value. |
| **Ablation** | The removal of a part of an activation, to see if an output changes. |
| **Activation patching** | An activation is copied from a run on one prompt into a run on another prompt. |
| **Causal tracing** | A patch for each layer and each token position, which shows where a piece of information is necessary. |
| **Circuit** | A small set of components, with their connections, that does one task. |
| **Attribution graph** | The circuit for one prompt: the nodes are features, and the edges show how much one feature causes the next. |
| **Logit lens** | The output matrix applied to an activation of a middle layer, to see which tokens this layer prefers. |
| **Faithfulness** | An explanation, or a chain of thought, is faithful when it gives the calculation or the reasons that really cause the output. |

## Control and oversight

These words are in the lesson [Control and oversight](safety/06-control.md).

| Word | Meaning |
|---|---|
| **Untrusted model** | The capable model that does the work and is possibly misaligned. |
| **Trusted model** | A weaker model that the developers assume to be safe. |
| **Control protocol** | The procedure that decides what occurs with each output of the untrusted model. |
| **Trusted monitoring** | The trusted model reads each output and gives it a suspicion score. |
| **Suspicion score** | The number that a monitor gives to an output: higher means more suspect. |
| **Audit budget** | The share of the outputs that the humans have time to check with care. |
| **Control evaluation** | A test that measures the safety of a protocol as a game between a red team and a blue team. |
| **Red team and blue team** | The blue team writes the protocol. The red team plays the untrusted model and attacks the protocol. |
| **Attack policy** | The instructions that replace the untrusted model and make it attack in the best way that the red team can find. |
| **Defer to trusted** | The protocol replaces the most suspect outputs by the answer of the trusted model. |
| **Resampling** | When a step is suspect, the protocol asks the untrusted model for new samples and runs the least suspect one. |
| **Untrusted monitoring** | A copy of the untrusted model is the monitor. |
| **Collusion** | Two copies of the untrusted model agree to let attacks pass. |
| **Honeypot** | An attack that the developers write to test a monitor. |
| **Chain of thought (CoT)** | The text in which a reasoning model writes its steps before the answer. |
| **CoT monitoring** | A monitor also reads the chain of thought. |
| **Monitorability** | How well a monitor can find a bad intent in the chain of thought. |
| **Obfuscated reasoning** | The model does not write its intent in its chain of thought. |
| **Scalable oversight** | Methods that let a weaker supervisor check a stronger system, also when the gap grows. |
| **Nested oversight** | A chain in which the guard checks a slightly stronger model, which then becomes the guard for the next model. |
| **Debate** | Two strong models argue for different answers, and a weaker judge selects one. |
| **Weak-to-strong generalization** | A strong model is trained on the labels of a weak supervisor. |
| **Human oversight** | Persons who can see what the system does, stop it and correct it. |
