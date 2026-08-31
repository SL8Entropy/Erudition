# Reading the Rocks with Machines

### A Complete Textbook on the ROGII "Wellbore Geology Prediction" Kaggle Competition
### From "what is a rock layer?" to the exact tricks used by the top five teams

*Written for a reader who knows one thing: that machine learning models can predict outputs from inputs.*

*Everything else — the geology, the drilling, the mathematics, the neural networks, the probability theory, and every technique in the five winning writeups — is built up from scratch in these pages.*

---

## How to use this book

This book has eight parts. They are meant to be read in order, but here is what each one does so you can skip back and forth once you have the map.

| Part | Title | What you get |
|---|---|---|
| **I** | The Ground Truth | The geology and the drilling. What rocks, wells, gamma rays, typewells and TVT actually are. No maths. |
| **II** | The Competition | The exact data, the exact task, the exact scoring rule, and why the task is genuinely hard. |
| **III** | Machine Learning From Zero | Losses, gradient descent, CNNs, U-Nets, attention, validation, boosting, ensembling. Everything the five teams assume you know. |
| **IV** | Guessing Where You Are | Probability, hidden states, HMMs, particle filters, dynamic programming. The "physics model" half of the competition. |
| **V** | Turning Geology Into a Picture | The single most important idea in the competition: the 2D matching canvas. Plus output heads and decoding. |
| **VI** | Making Fake Wells | Synthetic data generation, why it was essential, and the strange "realism paradox" every top team hit. |
| **VII** | The Five Solutions | One chapter per team. Every single technique in their writeups, explained. |
| **VIII** | Lessons | What all five agreed on, what nobody could make work, and how to study this yourself. |

**A note on difficulty.** I have marked a few sections with 🔬. Those are the places where I go a little deeper than a first read needs. Skip them the first time through; come back when you are dissecting the solutions in Part VII.

**A note on notation.** I use consistent symbols throughout. The full symbol table is in Appendix A, and there is a glossary of every domain term in Appendix B. If a word appears that you do not know, it is in Appendix B.

**A note on the code.** Code snippets are short, self-contained, and illustrative rather than production-grade. They use `numpy` and `torch`. Their job is to make an idea concrete, not to be a solution. Every one of them is a "toy" you should be able to run and poke.

## Table of contents

**PART I — THE GROUND TRUTH**

- [Chapter 1. Rocks come in layers](#chapter-1-rocks-come-in-layers)
- [Chapter 2. Drilling a well: three kinds of depth](#chapter-2-drilling-a-well-three-kinds-of-depth)
- [Chapter 3. The gamma ray log: the barcode reader](#chapter-3-the-gamma-ray-log-the-barcode-reader)
- [Chapter 4. The typewell: the reference barcode](#chapter-4-the-typewell-the-reference-barcode)
- [Chapter 5. TVT, precisely](#chapter-5-tvt-precisely)
- [Chapter 6. Geosteering: the human job being automated](#chapter-6-geosteering-the-human-job-being-automated)
- [Chapter 7. Faults, dip changes, and why wells go catastrophically wrong](#chapter-7-faults-dip-changes-and-why-wells-go-catastrophically-wrong)

**PART II — THE COMPETITION**

- [Chapter 8. The data](#chapter-8-the-data)
- [Chapter 9. The task, stated three ways](#chapter-9-the-task-stated-three-ways)
- [Chapter 10. The metric: pooled RMSE, and what it does to your strategy](#chapter-10-the-metric-pooled-rmse-and-what-it-does-to-your-strategy)
- [Chapter 11. Public leaderboard, private leaderboard, and the shake-up](#chapter-11-public-leaderboard-private-leaderboard-and-the-shake-up)
- [Chapter 12. Why this problem is hard: a catalogue](#chapter-12-why-this-problem-is-hard-a-catalogue)

**PART III — MACHINE LEARNING FROM ZERO**

- [Chapter 13. What a model actually is](#chapter-13-what-a-model-actually-is)
- [Chapter 14. Regression, classification, and the sneaky third option](#chapter-14-regression-classification-and-the-sneaky-third-option)
- [Chapter 15. Loss functions — the six that appear in this competition](#chapter-15-loss-functions-the-six-that-appear-in-this-competition)
- [Chapter 16. How the knobs get turned: gradient descent and its entourage](#chapter-16-how-the-knobs-get-turned-gradient-descent-and-its-entourage)
- [Chapter 17. Overfitting, and the whole toolbox against it](#chapter-17-overfitting-and-the-whole-toolbox-against-it)
- [Chapter 18. Trees, boosting, and the tabular models](#chapter-18-trees-boosting-and-the-tabular-models)
- [Chapter 19. Validation — the chapter that decided this competition](#chapter-19-validation-the-chapter-that-decided-this-competition)
- [Chapter 20. Neural networks, built up piece by piece](#chapter-20-neural-networks-built-up-piece-by-piece)
- [Chapter 21. Ensembling: the last 0.5 RMSE](#chapter-21-ensembling-the-last-05-rmse)
- [Chapter 22. A vocabulary checkpoint](#chapter-22-a-vocabulary-checkpoint)

**PART IV — GUESSING WHERE YOU ARE**

- [Chapter 23. Probability, in the amount you need](#chapter-23-probability-in-the-amount-you-need)
- [Chapter 24. State-space models: the shape of the problem](#chapter-24-state-space-models-the-shape-of-the-problem)
- [Chapter 25. Hidden Markov Models](#chapter-25-hidden-markov-models)
- [Chapter 26. Particle filters](#chapter-26-particle-filters)
- [Chapter 27. Dynamic programming and exact expectation decoding](#chapter-27-dynamic-programming-and-exact-expectation-decoding)
- [Chapter 28. 🔬 Dynamic Time Warping, in one page](#chapter-28-dynamic-time-warping-in-one-page)

**PART V — TURNING GEOLOGY INTO A PICTURE**

- [Chapter 29. The problem with the obvious approach](#chapter-29-the-problem-with-the-obvious-approach)
- [Chapter 30. The matching canvas](#chapter-30-the-matching-canvas)
- [Chapter 31. The channels: what everyone actually puts in the picture](#chapter-31-the-channels-what-everyone-actually-puts-in-the-picture)
- [Chapter 32. Output heads: five ways to say "the path is here"](#chapter-32-output-heads-five-ways-to-say-the-path-is-here)
- [Chapter 33. Decoding, drift, and re-anchoring](#chapter-33-decoding-drift-and-re-anchoring)
- [Chapter 34. Multimodality, one more time — the intellectual centre](#chapter-34-multimodality-one-more-time-the-intellectual-centre)

**PART VI — MAKING FAKE WELLS**

- [Chapter 35. Why it works, and the one assumption it rests on](#chapter-35-why-it-works-and-the-one-assumption-it-rests-on)
- [Chapter 36. Where the pieces come from](#chapter-36-where-the-pieces-come-from)
- [Chapter 37. The realism paradox](#chapter-37-the-realism-paradox)
- [Chapter 38. Staging: pretrain, post-pretrain, fine-tune](#chapter-38-staging-pretrain-post-pretrain-fine-tune)

**PART VII — THE FIVE SOLUTIONS**

- [Chapter 39. First place — Ruby: "2D alignment"](#chapter-39-first-place-ruby-2d-alignment)
- [Chapter 40. Second place — Bilzard: AnchorCNN and conditional probabilistic path modelling](#chapter-40-second-place-bilzard-anchorcnn-and-conditional-probabilistic-path-modelling)
- [Chapter 41. Third place — Takoi & tereka: five candidates and a gate](#chapter-41-third-place-takoi-tereka-five-candidates-and-a-gate)
- [Chapter 42. Fourth place — James, Lightsource, Alijs & Arunodhayan: three pipelines, one blend](#chapter-42-fourth-place-james-lightsource-alijs-arunodhayan-three-pipelines-one-blend)
- [Chapter 43. Fifth place — daimaru: synthetic-data-centric CNN](#chapter-43-fifth-place-daimaru-synthetic-data-centric-cnn)
- [Chapter 44. All five, side by side](#chapter-44-all-five-side-by-side)

**PART VIII — LESSONS**

- [Chapter 45. The transferable lessons, ranked](#chapter-45-the-transferable-lessons-ranked)
- [Chapter 46. A study plan](#chapter-46-a-study-plan)
- [Chapter 47. Writing the research paper](#chapter-47-writing-the-research-paper)

**APPENDICES**

- [Appendix A — Symbols](#appendix-a-symbols)
- [Appendix B — Glossary](#appendix-b-glossary)
- [Appendix C — The numbers, collected](#appendix-c-the-numbers-collected)
- [Appendix D — Sources](#appendix-d-sources)

---

# PART I — THE GROUND TRUTH
## The geology and the drilling, with no mathematics at all

---

## Chapter 1. Rocks come in layers

If you dig straight down into the Earth almost anywhere that oil is found, you do not find a uniform lump of stone. You find **layers** — sheets of rock stacked on top of each other like the pages of a closed book, or like the layers of a very large, very old lasagne.

Each layer was laid down at a different time, in a different environment. A layer of **sandstone** was once a beach or a river delta: grains of sand settled, got buried, got squeezed for a hundred million years, and became rock. A layer of **shale** was once deep still water: fine mud and clay settled slowly out of it. A layer of **limestone** was once a warm shallow sea full of shells and coral.

Three facts about these layers drive this entire competition:

**Fact 1 — Layers are laid down in a fixed order and that order never changes.**
Because sediment settles from above, the layer at the bottom is always older than the layer above it. Geologists call this the *principle of superposition*, and it is the oldest idea in geology. If you know that in this part of the world the sequence going down is *chalk → shale → limestone → shale → sandstone*, then that is the sequence everywhere in this part of the world. You may find the layers at different depths in different places. You may find some of them thicker or thinner. But you will not find them shuffled.

This is why the whole problem is solvable at all. The rock column is a **barcode**, and it is the same barcode everywhere in the field.

**Fact 2 — Layers are not flat.**
Over geological time the whole stack gets tilted, bent, folded, and pushed around. So a layer that was originally horizontal now slopes gently. The angle of that slope is called the **dip**. In the oilfield the dip is usually small — a fraction of a degree to a few degrees — but "small" over a two-mile-long well adds up to a lot of vertical movement.

Dip is also not perfectly constant. It changes slowly across a field. It can also change abruptly at a **fault**, which is a crack where one side of the rock has slipped past the other. At a fault, the barcode jumps: you are suddenly in a different part of the sequence than you were a foot ago.

**Fact 3 — Different layers are physically different, and you can measure that difference.**
This is the hinge on which everything turns, and it is Chapter 3.

---

## Chapter 2. Drilling a well: three kinds of depth

### 2.1 The well itself

Historically, wells were drilled straight down. Modern oil and gas wells — especially in shale plays — are drilled in an L shape:

```
             surface
   ┌────────────┬─────────────────────────────────────────────►
   │            │
   │  VERTICAL  │
   │  SECTION   │
   │            │
   │            ╲
   │             ╲   ← "the build" / "the curve"
   │              ╲
   │               ╲______________________________________
   │                                                       ► toe
   │                THE LATERAL (horizontal section)
   ▼ depth
```

- The **vertical section** goes straight down for a mile or two.
- The **build** or **curve** turns the well from vertical to horizontal over a few hundred feet.
- The **lateral** (also "the horizontal") then runs sideways, often for one to three miles, staying inside a single thin target layer of rock.

Why? Because the valuable rock — the shale that holds the oil — might be only 100 feet thick. A vertical well passes through 100 feet of it and stops. A horizontal well runs *along* it for 10,000 feet and contacts a hundred times more of it. That is the entire economic reason horizontal drilling exists.

But it creates a problem: **you have to stay inside the layer.** The layer is thin and it is tilted, and you cannot see it. Solving that problem in real time is called **geosteering** and it is Chapter 6.

### 2.2 Three completely different meanings of "depth"

This is the number one source of confusion for newcomers, so we will be very explicit. There are three quantities, and mixing them up makes the whole competition incomprehensible.

**MD — Measured Depth.**
How much pipe you have put in the hole. It is distance measured *along the borehole*, from the surface, following every twist and turn. If your well goes 8,000 ft down and then 10,000 ft sideways, the toe of the well is at MD = 18,000 ft.

MD is the natural "clock" of a well. Every measurement, every log reading, every survey point is stamped with an MD. Think of MD as **time** or as **position along the well**. In this competition, MD is the horizontal axis of everything.

**TVD / Z — True Vertical Depth.**
How far below the surface (or below sea level) you actually are, straight down. In the example above, at MD = 18,000 ft the TVD is still about 8,000 ft, because the last 10,000 ft went sideways, not down.

In the competition data this is given as the **Z** coordinate, alongside **X** and **Y** (the map position — how far north and how far east). X, Y and Z are known at *every point of every well, including the part you have to predict.* This turns out to be enormously important.

> **Remember this:** you always know exactly where the drill bit is in space. What you don't know is *what rock it is in*.

**TVT — True Vertical Thickness.**
This is the one that matters and the one that has to be predicted. TVT does not tell you where you are in the *Earth*. It tells you where you are in the **rock column** — which page of the book you are on.

Formally, TVT is a vertical thickness measured from a reference surface (a "marker" — the top of some recognisable layer) down (or up) to the point of interest. Practically, in this competition:

> **TVT = your position inside the layered rock sequence, expressed as a vertical distance from a fixed marker.**

Two points with the same TVT are in the **same rock layer**, even if they are miles apart and at completely different TVDs. Two points with the same TVD may be in totally different layers, if the rocks dip between them.

### 2.3 An analogy that will carry you a long way

Imagine an enormous, very slightly tilted multi-storey car park. It has 40 floors. You are driving a car through it in total darkness.

- **MD** = the odometer of your car. How far you have driven.
- **X, Y, Z** = a perfect GPS. You know exactly where you are in 3D space, always.
- **TVT** = *which floor of the car park you are currently on.*

The catch is that the car park is tilted, so knowing your altitude (Z) does not tell you your floor. As you drive north, the floors slope upward beneath you; if you drive perfectly level you will slowly pass from floor 12 down to floor 11 to floor 10, without changing altitude at all.

TVT is your floor number. Predicting TVT is figuring out which floor you are on, in the dark. And Chapter 3 gives you the only sensor you have.

### 2.4 Two more angles you will see

- **Inclination (or pitch)** — how tilted the borehole is. 0° = straight down, 90° = perfectly horizontal. In the lateral it hovers near 90°, wobbling a degree or two either side. That wobble is why Z changes slowly along a lateral rather than staying perfectly flat.
- **Azimuth** — the compass direction the well is heading. North, north-east, etc.

Azimuth matters because dip has a direction. If the rocks tilt down toward the south-east, then a well drilled to the south-east goes **downdip** (layers fall away below you) and a well drilled north-west goes **updip** (layers rise toward you). Two wells in the same field with opposite azimuths will see the layer sequence run in opposite directions. Several teams in this competition fed azimuth (usually as its sine and cosine, so that 359° and 1° are close together) directly into their models for exactly this reason.

---

## Chapter 3. The gamma ray log: the barcode reader

### 3.1 What a "log" is

As the drill bit cuts forward, instruments in the drill string measure properties of the rock and send them to the surface. A recorded curve of "measurement vs. depth" is called a **well log**. When it is measured while drilling it is called **LWD** — Logging While Drilling.

There are many kinds of logs — resistivity, density, sonic, neutron porosity. This competition uses exactly one, the most universal and the cheapest: **gamma ray**.

### 3.2 What gamma ray measures

Rocks are naturally, faintly radioactive. Certain elements — potassium-40, thorium, uranium — emit gamma radiation as they decay. These elements concentrate in **clay minerals**.

So:

- **Shale and mudstone** are made of clay → lots of potassium and thorium → **high gamma ray**.
- **Clean sandstone, limestone, chalk, salt** have little clay → **low gamma ray**.

The gamma ray tool is just a scintillation counter that counts gamma photons per second and reports the result in **API units** (a standardised scale defined by a calibration pit in Houston). Typical values run from about 10 API (clean limestone) to 200+ API (organic-rich shale).

### 3.3 Why this makes a barcode

Because the layers are in a fixed order, and each layer has its own characteristic gamma ray level, the gamma ray curve plotted against depth in a vertical well is a **signature of the rock column**. It has recognisable features: a sharp spike here, a smooth ramp there, a distinctive "double hump", a low flat interval.

Geologists have used this for a century. They call it **correlation**: you take the gamma ray curve from one well, slide it up and down against the curve from another well, and find where the patterns line up. When they line up, you know which parts of the two wells are in the same rock.

That sliding-and-matching operation is *precisely* the task in this competition, done ten million times by a computer.

```
   TYPEWELL (vertical)            HORIZONTAL WELL (lateral)
   GR →                           GR along MD →
   ┌──────┐                       
 T │   ▐  │  low                  ─────────────────────────────►  MD
 V │  ▐▐▐ │  spike                     ╱╲    ╱╲╲      ╱╲
 T │ ▐    │  low                      ╱  ╲__╱   ╲____╱  ╲___
   │▐▐▐▐▐ │  HIGH  (thick shale)  
 ↓ │  ▐   │  low                   "which part of the left-hand
   │ ▐▐   │  medium                 barcode am I sitting in,
   └──────┘                        at each point along MD?"
```

### 3.4 Two crucial complications

**Complication 1: the resolution difference.**
In a *vertical* well the tool moves straight down through the layers, so a 1-foot-thick layer is sampled over 1 foot of hole. In a *horizontal* well running nearly parallel to the layers, the tool spends a hundred feet of hole inside that same 1-foot layer. The horizontal log therefore has a *far* finer effective vertical resolution — it stretches thin beds out over long distances.

The competition host pointed this out explicitly in the forums: the lateral gamma ray recorded *before* the prediction point is a **better** vertical reference than the typewell itself, because it is a higher-resolution look at the same rocks. Multiple top teams (2nd place channel `x6`, 3rd place "self-reference", 5th place "type well correction") built features around exactly this.

The 5th-place solution took the observation further: the typewell log is effectively a **smoothed** version of the true rock column, because thin beds get blurred out by the tool's vertical resolution. That insight caused a complete redesign of their synthetic data generator (Chapter 36.4).

**Complication 2: the same rocks do not give the same number twice.**
Different tools, different mud, different hole conditions, different logging companies, different years. So the same layer measured in two wells might read 90 API in one and 105 API in the other. This offset is called **GR bias** or **GR shift**. It is one of the main things that makes naive pattern matching fail, and several solutions carry an explicit, slowly-drifting "GR bias" as part of their internal state (3rd place HMM and particle filter both do).

Also: the gamma ray is sometimes simply **missing**. Tool failures, gaps in transmission, sections where nothing was recorded. Every team had to handle NaNs, and several made a virtue of it by *deliberately masking* gamma ray during training (Chapter 21) so the model would learn to survive without it.

---

## Chapter 4. The typewell: the reference barcode

A **typewell** (also written "type well", and elsewhere called a "type log" or "offset well") is a nearby, usually **vertical**, well that has already been drilled and fully interpreted. For a typewell we know:

- its gamma ray curve, top to bottom;
- the TVT of every point on it (by definition — a vertical well passes straight through the layers, so its depth *is* essentially its stratigraphic position);
- the names and positions of the geological layers ("geology" column: formation tops like **BUDA**, **ANCC**, etc. — these are the expert-picked marker surfaces that define the local rock column).

So the typewell provides the function

> **f(TVT) = expected gamma ray value at stratigraphic position TVT**

This function `f` is the reference barcode. Every single approach in the top five uses it, one way or another. The whole task can be restated in one line:

> **Given the observed gamma ray along the lateral, and given the reference barcode f, find the path TVT(MD) that best explains what was observed.**

Each horizontal well in the competition is assigned a typewell. Many horizontal wells **share** a typewell — they are "siblings". Both the 3rd-place and 5th-place teams exploited this hard: if five laterals share a typewell, you can pool their gamma ray data to build a much better reference curve than the typewell alone gives you. The 2nd-place team went further and discovered that the 773 training typewells actually collapse into just **54 "master" series** — long continuous gamma-ray-vs-depth logs from which every individual typewell is just a cropped window. That discovery is what made their synthetic data generator possible (Chapter 36.1).

---

## Chapter 5. TVT, precisely

Now we can be exact, because this is the target variable.

### 5.1 The master identity

Let:

- `z` = the vertical position of the drill bit (the Z coordinate; known everywhere, always);
- `z_layer` = the vertical position of the **rock marker surface** at the bit's map location (unknown);
- `TVT` = the bit's stratigraphic position.

These three are related by a purely geometric identity. Up to a per-well constant `b`:

```
    z_layer  =  TVT + z − b            ⟺        TVT  =  z_layer − z + b
```

Differentiating along the well (i.e. looking at how things *change* from one row to the next):

```
    ΔTVT  =  Δz_layer  −  Δz                                        ... (★)
```

**This is the single most important equation in the competition.** The 2nd-place team put it at the very top of their writeup and called it "the key insight from the data". The 4th-place team's James expressed it in words: *"Change in TVT = change in rock elevation − change in the drill's Z coordinate."*

(Different companies use different sign conventions for whether depth increases up or down; do not worry about the sign. What matters is the structure.)

### 5.2 Why (★) is so powerful

Read (★) out loud in plain English:

> **Your position in the rock column changes for exactly two reasons: either you moved up or down, or the rocks moved up or down beneath you.**

Now: **`Δz` is completely known.** You are given X, Y, Z at every point of every well, including the part you have to predict. So half of (★) is handed to you for free.

That means the *entire* unknown is `Δz_layer` — the **structural slope**, how the rock surface itself rises and falls. And structural slope is a *geological* quantity: it is smooth, it is nearly constant over long stretches, it varies slowly across a field, and neighbouring wells share it.

The 2nd-place team examined the expert-picked formation surface columns in the training data and found that `dz_layer/dMD` is **staircase-shaped**: constant over most of a well, with occasional steps (faults or dip changes), and steps occur in only about 10% of cases. So for most of a well, `ΔTVT` should look *exactly* like `−Δz` — the two curves run parallel.

This reframes the problem enormously:

| Naive framing | Framing via (★) |
|---|---|
| Predict a 10,000-ft-long wiggly curve TVT(MD) | Predict a nearly-constant number: the structural slope |
| Model must invent everything | Model gets the wiggles for free from Z, and only has to estimate a smooth trend |

Every top-five solution exploits this, though they do it in different ways:

- **1st place** feeds `z_diff` (the change in Z) as an input channel and predicts *accumulated* TVT.
- **2nd place** feeds `dip z` — the per-column slope of Z — as input channel `x8`, and keeps `z_layer` (not `z`) as its internal representation so that every data augmentation automatically stays physically consistent with (★).
- **3rd place** makes "formation rate" `d(TVT + Z)/dMD` — which is precisely `dz_layer/dMD` — an explicit part of the hidden state of their HMM and particle filter.
- **4th place** (James) makes it the *entire first stage*: predict the slope `dz_layer/dMD` with tabular regression models, integrate it to get a baseline TVT path, and only then use vision models to correct that path.
- **5th place** feeds "cumulative −ΔZ" as channel 5 — literally "where the path would sit under a flat-layer assumption".

If you take one thing from Part I, take equation (★).

### 5.3 The two forces, drawn

```
 Case A: flat layers, wobbling well          Case B: level well, dipping layers
 
 ═══════════════════════ layer top           ══════════╗
                                                        ╚═══════════ layer top
     ╭─╮       ╭──╮                            ─────────────────────────  well
 ────╯ ╰───────╯  ╰────  well
 
 TVT changes because YOU moved              TVT changes because the ROCKS moved
 (this part is KNOWN from Z)                (this part is the UNKNOWN slope)
```

---

## Chapter 6. Geosteering: the human job being automated

Here is what actually happens on a rig at 3 a.m.

A geologist sits in a control room, often hundreds of miles away, watching data stream in. The drill bit is 12,000 feet along a lateral, inside a target zone perhaps 30 feet thick. Every few minutes, a new stretch of gamma ray arrives.

The geologist has, on one screen, the typewell gamma ray plotted vertically. On the other, the incoming lateral gamma ray plotted against MD. Their job is to answer one question, continuously:

> *"Given what the gamma ray just did, where am I in the layer stack, and am I drifting out of the target?"*

They do this by **correlation**: they look at the shape of the recent lateral gamma ray, find the piece of the typewell curve it looks like, and thereby infer their stratigraphic position. They draw a picture — a cross-section showing the layers dipping across the well path — and they update it every hour. When they conclude the well is drifting up out of the zone, they call the directional driller and say "drop 0.4 degrees".

Their output *is* the TVT curve. The "ground truth" labels in this competition are exactly these expert interpretations.

Three consequences of that fact, which the top teams all noticed:

1. **The labels are the opinions of humans.** They are not measured. Where the data was ambiguous, an expert made a judgement call. Another expert might have called it differently. The 1st-place solution ran into a case where a feature consistently improved local cross-validation but hurt the public leaderboard, investigated exhaustively, and concluded: *"I attribute the discrepancy to inconsistent labels and chose to trust the local CV."*

2. **The labels are relative, not absolute.** The 2nd-place team tried modelling the formation surfaces directly from neighbouring wells and found it could not fix the catastrophically bad wells. Their diagnosis: *"the teacher labels (expert interpretations) do not reference the formations directly — they are relative judgements based on GR matching."* The expert is matching squiggles, not consulting a regional structure map. So a model that consults a regional structure map is solving a subtly different problem than the one being scored.

3. **The task is genuinely, irreducibly ambiguous.** Gamma ray barcodes repeat. A stack of similar-looking shale beds gives you three or four places where the recent squiggle fits about equally well. The expert picks one. A model that is forced to pick one will pick wrong sometimes. A model that can hold several hypotheses at once does better. This is the **multimodality** problem and it is the intellectual centre of the competition — Chapters 30 and 34.

---

## Chapter 7. Faults, dip changes, and why wells go catastrophically wrong

Most of a lateral is boring: constant dip, smooth gamma ray, the model tracks it fine. The score is not decided there.

The score is decided at the **breaks**:

- **Faults.** The rock is cut and offset. In one foot of MD, your TVT jumps by 5, 20, 50 feet. Any model that assumes smooth motion will miss this entirely and then be wrong for the remaining 8,000 feet of the well.
- **Dip changes.** The structural slope changes. If your model has locked onto the old slope, it now drifts steadily away from truth, and the error accumulates linearly with distance.
- **Mis-correlation / layer aliasing.** The model matches the gamma ray to the *wrong* repeat of a similar-looking pattern. It is now confidently tracking a layer 30 feet away from where it actually is, and the gamma ray keeps "confirming" this because that layer looks similar. This is the failure mode 4th place's Alijs described as *"the wrong-layer matches that hurt simpler particle-filter solutions."*

Because the scoring metric squares the errors and pools every row of every well (Chapter 10), a handful of wells with catastrophic mis-correlations dominate the entire leaderboard. The 2nd-place writeup states this as their central strategic observation:

> *"The metric moves far more when you fix a few badly failing wells than when you further improve wells that are already doing fine."*

Their whole model-selection procedure (Chapters 19.6 and 40.9) was designed around it.


---

# PART II — THE COMPETITION
## The data, the task, the score, and why it is hard

---

## Chapter 8. The data

**Host:** ROGII, a company that makes geosteering software (their product `StarSteer` is used for exactly the manual workflow described in Chapter 6). **Platform:** Kaggle. **Ran:** mid-2026, ending early August 2026.

### 8.1 What you are given

**For each horizontal well** (one file per well, one row per foot of MD):

| Column | Meaning | Known at test time? |
|---|---|---|
| `MD` | measured depth along the borehole, 1 ft steps | ✅ always |
| `X`, `Y` | map coordinates (easting, northing) | ✅ always |
| `Z` | vertical coordinate of the bit | ✅ **always — including after PS** |
| `GR` | gamma ray reading (may be NaN) | ✅ always |
| `TVT_input` | the TVT values, **but only up to the PS point** | ✅ up to PS, blank after |
| `TVT` | the true answer | ❌ training only |

**For each typewell** (a vertical reference well; several laterals may share one):

| Column | Meaning |
|---|---|
| `TVT` | stratigraphic position, top to bottom |
| `GR` | gamma ray at that position |
| `geology` | formation/layer names at their tops (e.g. `BUDA`, `ANCC`) |

### 8.2 The PS point

**PS = Prediction Start.** Somewhere along the lateral there is a line. Before it, the expert's TVT interpretation is handed to you. After it, it is hidden and you must predict it.

```
   MD ───────────────────────────────────────────────────────────────►
        vertical    curve       LATERAL
   │─────────────│────────│──────────────┬───────────────────────────┤
                                         PS
        TVT_input given ───────────────► │ ◄───── TVT hidden: PREDICT
        GR, X, Y, Z given ───────────────┼───────────────────────────►
                                    (still given!)
```

The stretch before PS is called the **known prefix** (or "pre-PS interval"). It is enormously valuable and every top team squeezed it:

- it tells you the well's **current** stratigraphic position — an anchor point;
- it tells you the well's **current dip** — the recent slope of `TVT + Z`;
- it gives you a high-resolution, *same-tool, same-well* gamma-ray-vs-TVT reference (Chapter 3.4), better calibrated to this well than the typewell is.

The known-prefix fraction varies between wells. 4th place's Lightsource randomised it during training between roughly 25% and 65% so that their model would be robust to any prefix length.

### 8.3 Scale

- **773 training wells**, **3,783,989 training rows** (so ~4,900 rows ≈ 4,900 ft of well each, on average).
- **~200 test wells**, split into a public leaderboard subset of roughly **50–60 wells** and a private subset of **148 wells**.
- Typewells consolidate into ~**54 master series** (2nd place's finding).
- It was a **notebook-only** competition: your submission is code that runs inside a Kaggle notebook against the hidden test data, on a **T4 GPU**, with a runtime limit. This is why every solution talks about inference cost — 2nd place notes their full 15-checkpoint × 8-TTA ensemble scored 200 wells in about 40 minutes.

---

## Chapter 9. The task, stated three ways

**Plain English.** For each horizontal well, predict the stratigraphic position of the drill bit at every foot after the prediction-start point.

**As a function.** Learn a mapping

```
   (MD, X, Y, Z, GR)_lateral  ,  (TVT, GR)_typewell  ,  TVT_input[:PS]
        ⟼   TVT[PS:]
```

**As an alignment.** For every MD position after PS, choose a row of the typewell. That is, find a monotone-ish path through a 2D grid whose axes are (MD, TVT), maximising agreement between observed lateral gamma ray and typewell gamma ray at the chosen TVT, subject to the path being physically smooth.

The third framing is the one the winners used. Hold onto it — Part V is entirely about it.

---

## Chapter 10. The metric: pooled RMSE, and what it does to your strategy

### 10.1 The definition

For every predicted row `i` (across all test wells, pooled together into one big list), let `e_i = TVT_pred,i − TVT_true,i`. The score is

```
                       ┌────────────────────┐
                       │  1   N             │
    RMSE   =    sqrt   │ ───  Σ   e_i²      │
                       │  N  i=1            │
                       └────────────────────┘
```

Lower is better. The winning private score was **5.639 ft**. Typical strong solutions sat around 5.8–6.2 ft. A pure particle filter got around 6.7–7.0 ft. So the *entire* competitive range — from "decent baseline" to "1st place" — was about one foot of RMSE.

Note the word **pooled**. It is not "average the RMSE of each well". Every row of every well goes into one pool. So:

- **long wells count more** than short wells (more rows);
- **squaring** means a well that is off by 40 ft contributes 1,600 per row, while a well off by 2 ft contributes 4 per row — a 400× difference.

### 10.2 Consequence 1: catastrophes dominate everything

Suppose 90% of your wells have a 3-ft RMSE and 10% have a 15-ft RMSE. Pooled:

```
    sqrt(0.9 × 3² + 0.1 × 15²)  =  sqrt(8.1 + 22.5)  =  5.53
```

The 10% of bad wells contribute **73%** of the squared error. Improving the good wells from 3 ft to 2.5 ft moves the score to 5.42. Fixing *one third* of the bad wells (from 15 to 10) moves it to 4.85. Three times the gain, for a much smaller-sounding change.

This is why 2nd place wrote their entire model-selection protocol around "is this improvement real, or did I just get lucky on five wells?" (Chapters 19.6 and 40.9), and why 1st place spent so much effort on augmentations that simulate rare events like fault jumps.

### 10.3 Consequence 2: hedge your bets — predict the *average* of the hypotheses

This one is subtle and it is the deepest strategic idea in the competition.

Suppose at a given point the gamma ray is genuinely ambiguous. There is a 50% chance you are at TVT = +10 and a 50% chance you are at TVT = −10.

| Strategy | If truth is +10 | If truth is −10 | Expected squared error |
|---|---|---|---|
| Commit to +10 | 0 | 400 | **200** |
| Commit to −10 | 400 | 0 | **200** |
| Predict 0 (the mean) | 100 | 100 | **100** |

**Predicting the average of the hypotheses — a path that is physically impossible, that corresponds to no real geology at all — halves the expected squared error.**

This is just the fact that the mean minimises squared error, but the consequence for design is enormous:

- 2nd place found that **Viterbi decoding (the single most likely path) was consistently worse than expectation decoding (the probability-weighted average path)** — and built their whole decoder around exact expectation via dynamic programming.
- 4th place's James wrote: *"The competition scoring metric rewards models for predicting paths between the likely valid possibilities."* Their model outputs three candidate trajectories plus probabilities and returns the weighted average.
- 2nd place also observed the flip side: *"a model that regresses a single teacher path indeed collapses onto the average of these modes"* — but a naively-trained regressor collapses onto the average **badly**, because it never learns that there were separate modes at all. The trick is to model the modes explicitly and *then* average them, which gives you both a better average and, as a bonus, an uncertainty estimate.
- 4th place's Arunodhayan measured it directly: switching from "zero-crossing decode" (commit to the boundary) to "soft expectation decode" was worth **−1.70 RMSE**. That is by far the largest single improvement reported anywhere in the five writeups.

> **Law of this competition:** *keep the ambiguity alive as a probability distribution for as long as possible, and only collapse it to a number at the very last step, by taking an expectation.*

---

## Chapter 11. Public leaderboard, private leaderboard, and the shake-up

Kaggle splits the test set. During the competition you see your score on a **public** subset (~52 wells here). At the end you are ranked on the **private** subset (148 wells), using two submissions you nominate in advance.

With only ~50 wells and a metric dominated by catastrophes, the public leaderboard here was **noise**. The writeups are unusually blunt about it:

- **5th place** listed their last six submissions. Their best public score was their *worst* private score, and their worst public was their second-best private. They concluded: *"Public and private are almost anti-correlated, while CV and private rank-correlate very well."*
- **1st place** found a feature family (XY-neighbour information) that reliably improved local CV by 0.3 RMSE but hurt the public LB. They ran five different diagnostic statistics trying to find a legitimate reason, found none, and chose to trust CV. They won.
- **3rd place** explicitly listed "OOF-only blending — final weights and model-selection decisions are determined only from OOF predictions, never by fitting to the Public LB" as a design rule.
- **4th place** ended up in a curious position: their ensemble's three members scored 5.730 / 7.045 / 9.138 on private — two of the three were near-useless there — but the *blend* held up at 5.870 and took a gold.

**The lesson, which is a general Kaggle lesson but was extreme here:** build a validation scheme you trust, then trust it. Chapter 19 covers how, and Chapter 19.6 covers the most sophisticated version anyone in this competition built.

### 🔬 11.1 A distribution shift that nobody fully explained

5th place ran an unusually clean experiment. Their model trained on **synthetic wells only**, never fine-tuned on real data, scored:

| Model | CV | Public | Private |
|---|---|---|---|
| synthetic pretraining only | 6.250 | 7.186 | **6.342** |
| + fine-tuned on real wells | 4.844 | 5.600 | **5.835** |

Fine-tuning on the real training wells improved CV and public by about **1.5 ft**, but improved private by only about **0.5 ft**. Their reading: *"The training data and the public test data appear to be very similar, while the private test data seems to have a slightly different distribution... It looks like the competition ultimately tested robustness to this kind of small distribution shift."*

They noticed the test wells appeared skewed to the north-east of the field, and that restricting their CV to the north-eastern part of the training data reproduced their public score. The host replied that the split was random and the field is small. The question was never fully resolved — which is itself a good lesson about how hard it is to diagnose distribution shift from the outside.

---

## Chapter 12. Why this problem is hard: a catalogue

Before we build anything, let us be honest about the obstacles. Every technique in Parts V–VII exists to attack one of these.

**H1 — Ambiguity / multimodality.** The gamma ray barcode has repeating features. Several TVT hypotheses fit the recent data about equally well. → *Attacked by:* probability fields instead of point predictions, MTP loss, particle filters, HMM posteriors, expectation decoding.

**H2 — Error accumulation / drift.** Predict small step changes, integrate them over 8,000 rows, and a tiny bias becomes a huge offset. → *Attacked by:* anchoring predictions to the last known TVT, re-anchoring during decoding, predicting absolute offsets alongside deltas, ensembling models with different drift behaviour.

**H3 — Tiny dataset.** 773 wells. A modern CNN has hundreds of millions of parameters. → *Attacked by:* massive synthetic pretraining (Part VI), heavy augmentation, small heads on big backbones, transfer learning from ImageNet, k-fold ensembling.

**H4 — Noisy, human, inconsistent labels.** See Chapter 6. → *Attacked by:* robust losses (Huber, Student-t), robust model selection, trusting CV, refusing to over-fit the leaderboard.

**H5 — Corrupted / missing measurements.** GR bias between wells, GR gaps. → *Attacked by:* explicit bias states, calibrating the typewell against the lateral, GR-affine augmentation (`GR' = a·GR + b`) to force shape-reliance over level-reliance, deliberate GR masking during training.

**H6 — Rare catastrophic events.** Faults, dip breaks, mis-correlations. → *Attacked by:* simulating fault jumps in augmentation and in synthetic data, allowing low-probability large jumps in particle filters, bidirectional decoding, ensembling.

**H7 — A tiny, misleading public leaderboard.** → *Attacked by:* fold-safe CV, multiple split patterns, leave-largest-contribution-out testing.

**H8 — Runtime limits.** Notebook-only, T4 GPU. → *Attacked by:* efficient backbones, coarse MD grids with sub-grid TTA to recover resolution, careful ensemble budgeting.


---

# PART III — MACHINE LEARNING FROM ZERO
## Everything the five writeups assume you already know

You said you know one thing: machine learning models predict outputs from inputs. This part turns that one sentence into everything you need. If you already know a chapter, skim it — but do read Chapter 15 (losses) and Chapter 19 (validation) carefully, because this competition turns on both.

---

## Chapter 13. What a model actually is

A **model** is a function with adjustable knobs.

```
    output  =  f(input ; θ)
```

`θ` (theta) is the collection of knobs — the **parameters** or **weights**. **Training** means: find the setting of `θ` that makes the outputs match the known answers on data you have. **Inference** means: freeze `θ` and run the function on new data.

That is all. A linear regression is a model with two knobs (`y = ax + b`). A ConvNeXt-Large is a model with 200 million knobs. The idea is identical.

**Supervised learning** — the only kind used in this competition — means you have pairs `(input, correct output)` to learn from. Here the input is a well's gamma ray + trajectory + typewell, and the correct output is the expert's TVT curve.

**Features** are the numbers you feed in. **Feature engineering** is the craft of computing *better* numbers from raw data before feeding them in. It is enormously important here: `Δz` and `|typewell_GR − horizontal_GR|` are not in the raw files, but a model given them learns far faster than one that has to discover them.

**Targets** (or labels) are the numbers you want out.

---

## Chapter 14. Regression, classification, and the sneaky third option

**Regression** predicts a number. "TVT at this row = 7.3 ft." Natural for this task.

**Classification** predicts which of `K` boxes something falls in, by outputting a probability for each box. "TVT is in bin 12 with probability 0.4, bin 13 with probability 0.35, bin 30 with probability 0.2..."

Here is the sneaky third option, and it is what the winners used:

> **Turn a regression problem into a classification problem, then convert the predicted probabilities back into a number by taking their expectation.**

Chop the possible TVT values into, say, 400 bins of 0.5 ft each. Ask the model to output a probability for each bin. Then report

```
    TVT_predicted  =  Σ_k  p_k · TVT_of_bin_k
```

Why bother, when you could have just regressed the number directly?

1. **It can represent ambiguity.** A regressor outputs one number and has no way to say "either +10 or −10". A distribution over bins says exactly that.
2. **You get uncertainty for free.** A wide distribution means "I'm not sure"; a sharp one means "I'm confident". That confidence can be fed to a downstream model (3rd place's SoftMax Gate does exactly this).
3. **The expectation is automatically the RMSE-optimal hedge** (Chapter 10.3).
4. **Cross-entropy loss on bins trains more stably** than squared error on a wildly multimodal target.

1st place: *"I formulate the problem as a 2D alignment task and use cross-entropy loss as the main objective."*
2nd place: 21 discrete move classes, cross-entropy, then exact expectation by DP.
3rd place's Adaptive 1D SDF: "Compare 256 TVT candidates... Apply SoftMax with temperature 1.75 and output the expected TVT."
5th place: "head2: row-classification logits".

Four of the five top solutions do this. It is not a coincidence.

```python
import numpy as np

# 400 bins, 0.5 ft each, centred on 0 (relative to the anchor)
bin_centres = (np.arange(400) - 200 + 0.5) * 0.5      # -99.75 ... +99.75 ft

logits = model_output                                  # shape (400,)
p = np.exp(logits - logits.max()); p /= p.sum()        # softmax
tvt_hat = float((p * bin_centres).sum())               # expectation decode

# bonus: free uncertainty
var = float((p * (bin_centres - tvt_hat)**2).sum())
sigma = var ** 0.5
```

---

## Chapter 15. Loss functions — the six that appear in this competition

The **loss** is a number that measures how wrong the model is. Training minimises it. Choosing the loss is choosing what "wrong" means, and it is one of the highest-leverage decisions you make.

### 15.1 MSE — Mean Squared Error

```
    L = mean( (y_true − y_pred)² )
```

The default for regression. Matches the competition metric directly (RMSE is just `sqrt(MSE)`, and minimising one minimises the other). Its defining property: **the value that minimises MSE is the mean.** That is the mathematical root of the hedging insight in Chapter 10.3.

Its weakness: because errors are squared, a single wildly wrong label yanks the model around. With human-interpreted labels (Chapter 6), that matters.

### 15.2 Huber loss — MSE that stops panicking

```
    L(e) =  0.5·e²            if |e| ≤ δ
            δ·(|e| − 0.5δ)    otherwise
```

Quadratic near zero, linear far out. So it behaves like MSE for ordinary errors but refuses to be dominated by outliers. `δ` (delta) is the changeover threshold.

Used by: 1st place (Huber on the expected TVT path), 4th place (Huber with thresholds tuned by Optuna — one model used δ = 9.48 ft, another δ = 0.549 ft, another δ = 3.013; the diversity of thresholds was itself used to diversify the ensemble).

```python
import torch, torch.nn.functional as F
loss = F.huber_loss(pred, target, delta=9.48)
```

### 15.3 Cross-entropy — the loss for "which bin?"

If the model outputs probabilities `p_k` over `K` bins and the truth is bin `t`:

```
    L = − log p_t
```

Confident and right → tiny loss. Confident and wrong → enormous loss. It is exactly the negative log-likelihood of the data under the model's distribution.

**Soft targets.** Instead of "all the mass on bin `t`", you can spread the target over neighbouring bins. Then

```
    L = − Σ_k  q_k · log p_k
```

where `q` is the target distribution. 1st place did precisely this: *"The training target is an exponentially smoothed probability distribution centred around the ground-truth alignment and normalised along the typewell dimension."* Smoothing the target tells the model "bin 13 is nearly right, bin 200 is very wrong", which a hard one-hot target cannot express — it treats being off by one bin and off by 200 bins as equally bad.

```python
import torch
K, sigma_bins = 400, 4.0
logits  = torch.randn(K)                      # model output, one logit per TVT bin
centres = torch.arange(K).float()
t = torch.tensor(197.0)                       # ground-truth bin (can be fractional)
q = torch.exp(-((centres - t).abs()) / sigma_bins)   # exponential smoothing
q = q / q.sum()
logp = torch.log_softmax(logits, dim=-1)
loss = -(q * logp).sum()
```

### 15.4 Gaussian NLL — predict the answer *and* your own uncertainty

Have the model output two numbers per point: a mean `μ` and a variance `σ²`. Then minimise the negative log-likelihood of a Gaussian:

```
    L = 0.5 · [ log(σ²) + (y − μ)² / σ² ]
```

Read it: the second term says "be accurate". The first term says "don't cheat by claiming infinite uncertainty". Together they force the model to report honest error bars: it is allowed to be inaccurate where the data is genuinely ambiguous, provided it *says* so.

3rd place used this for their main neural network and reported it beat plain RMSE loss (three-split ensemble: 5.4176 → 5.3645). Crucially, they did **not** use `σ` to change the prediction — they fed it to the ensemble gate as a confidence feature. 4th place's Arunodhayan used a Gaussian-NLL row loss too.

```python
import torch, torch.nn.functional as F
mu, raw_var = head(features).chunk(2, dim=-1)
var = F.softplus(raw_var) + 1e-3              # variance floor, keeps it positive
loss = 0.5 * (torch.log(var) + (y - mu)**2 / var)
loss = loss[valid_mask].mean()
```

### 15.5 Segmentation losses: BCE, Dice / Jaccard

When the model's output is an image of per-pixel probabilities ("is this pixel inside the region or not?"):

- **BCE** (binary cross-entropy) is per-pixel cross-entropy with two classes.
- **Dice / Jaccard (IoU)** losses measure region *overlap* rather than per-pixel correctness. They handle class imbalance well — if only 2% of pixels are "inside", BCE can be minimised by predicting "outside" everywhere, but Dice cannot.

4th place's Lightsource used `BCE + Dice/Jaccard + row_loss + optional smooth`. Their `row_loss` is interesting: it decodes the pixel probabilities into an actual TVT number and penalises *that* — so part of the loss is computed on the final answer, not just on the intermediate image. This "differentiable decode in the loss" pattern shows up again and again.

### 15.6 MTP — Multiple Trajectory Prediction loss

This one is specific to multimodal problems and comes from the Alyaev et al. paper on multi-modal inversion of geophysical logs, which 2nd, 4th and others cite.

The model outputs `M` candidate trajectories `T_1..T_M` and a probability `π_1..π_M` for each. The loss is:

```
    L_MTP  =  min over m of  Loss(T_m , y_true)      # only the BEST guess is penalised
              +  λ · CrossEntropy(π , argmin_m Loss(T_m, y))    # learn which is best
```

The magic is the `min`. Because only the closest candidate is penalised for accuracy, the candidates are free to spread out and cover different hypotheses instead of all collapsing onto the same average. Meanwhile the classification term teaches the model which candidate to trust.

**The catch**, which 4th place's James discovered and documented: pure MTP is optimal for "did we cover the truth?" but *not* for "what single number minimises RMSE?" So he trained on a **hybrid**: MTP loss + a loss on the probability-weighted average trajectory. He also found a failure mode when scaling up: past ~3 million training samples, ConvNeXt-Large started "disregarding the classification part of the MTP loss entirely in order to get lower weighted average regression loss" — the model found it could game the hybrid objective.

```python
import torch, torch.nn.functional as F
# preds: (B, M, T) candidate paths;  logits: (B, M);  y: (B, T)
per_mode = ((preds - y.unsqueeze(1))**2).mean(-1)      # (B, M)
best     = per_mode.argmin(dim=1)                      # (B,)
l_mtp    = per_mode.min(dim=1).values.mean()           # only the best is penalised
l_cls    = F.cross_entropy(logits, best)
pi       = logits.softmax(-1).unsqueeze(-1)
l_avg    = (((preds * pi).sum(1) - y)**2).mean()       # weighted-average path
loss     = l_mtp + 0.164 * l_cls + 1.0 * l_avg         # 4th place's Trial-1 weights
```

### 15.7 Others you will meet in passing

- **Smooth L1** — Huber with δ = 1. Used by 3rd place's SDF model.
- **Student-t likelihood** — a heavy-tailed alternative to the Gaussian, so single outlier readings do not dominate. 3rd place's HMM used a Student-t with **one degree of freedom** (i.e. a Cauchy) for its gamma ray emission model, explicitly *"making it more robust to outliers than a Gaussian likelihood."*
- **Rank-N-Contrast (RNC)** — a contrastive loss that teaches an embedding space to respect an *ordering*. Here: points whose TVTs are close should have similar embeddings; points far apart in TVT should be far apart in embedding space. 3rd place used it at weight 0.05 in their SDF model.

---

## Chapter 16. How the knobs get turned: gradient descent and its entourage

### 16.1 The core loop

The loss `L(θ)` is a landscape over the space of all possible weight settings. You want the bottom of a valley. You cannot see the landscape, but at your current position you can compute the **gradient** `∂L/∂θ` — the direction of steepest *increase*. So step the other way:

```
    θ  ←  θ  −  η · ∂L/∂θ
```

`η` (eta) is the **learning rate**: how big a step to take. Too big and you bounce out of the valley; too small and you take a million years.

**Backpropagation** is the algorithm that computes `∂L/∂θ` efficiently for a deep network. It is the chain rule from calculus, applied layer by layer from the output backwards. You will never implement it — `loss.backward()` does it — but you should know that is what the call means.

**Mini-batches.** Computing the gradient over all 3.8 million rows is slow, so you use a random subset (a *batch*) each step. Noisy, but much faster and the noise even helps escape bad valleys. One pass over the whole dataset is an **epoch**.

### 16.2 The optimisers you will see named

- **SGD** — plain gradient descent with mini-batches, usually with *momentum* (keep a running average of past gradients so you roll through small bumps).
- **Adam / AdamW** — the workhorse. Keeps per-parameter running averages of the gradient and of the squared gradient, and scales each parameter's step by its own history. AdamW is Adam with *decoupled weight decay* (see 17.2) and is the default for nearly all modern deep learning. Every solution in the top five uses AdamW or a variant.
- **Schedule-Free AdamW** — used by 3rd place. A recent variant that removes the need for a learning-rate schedule by using a particular averaging scheme internally. Convenient: one fewer thing to tune.

### 16.3 Learning-rate schedules

The learning rate is usually not constant.

- **Warmup** — start tiny and ramp up over the first few % of training. Large models are unstable at the start; warmup prevents an early explosion. 4th place used 10–30% warmup depending on the model.
- **Cosine annealing** — after warmup, decay `η` smoothly to near zero following a cosine curve. Big steps early to explore, tiny steps late to settle.
- **OneCycle** — warm up to a peak, then anneal down, and cycle the momentum in the opposite direction. 4th place used `OneCycleLR` with a peak LR of 2.053e-4 for pretraining.
- **WSD (Warmup–Stable–Decay)** — warm up, then hold constant for a long time, then decay at the end. 2nd place used *"a constant learning rate of 1e-3 after warmup (WSD schedule with no decay)"* — they skipped the decay phase entirely and relied on weight averaging instead.

### 16.4 EMA — exponential moving average of the weights

Instead of using the final weights, keep a slowly-updated shadow copy:

```
    θ_EMA  ←  α · θ_EMA  +  (1 − α) · θ
```

with `α` around 0.995–0.999. The EMA weights are a smoothed version of the training trajectory. They almost always generalise better than the raw final weights, because the raw weights are still bouncing around from the last few noisy batches. Free improvement; everyone uses it. 2nd place, 3rd place (EMA 0.995 for the NN, 0.999 for the SDF model) both report it.

```python
import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)
```

### 16.5 Mixed precision: FP32, FP16, BF16

Neural networks do not need full 32-bit precision. Training in 16-bit halves the memory and roughly doubles the speed on modern GPUs.

- **FP16** has a narrow exponent range and overflows easily; it needs "loss scaling" to work.
- **BF16** (bfloat16) has the *same exponent range as FP32* with fewer mantissa bits. Less precise per number, but almost never overflows. This is why it is preferred for large models.

1st place hit this directly: they replaced LayerNorm with BatchNorm in ConvNeXt because it *"consistently works better, although BF16 training is required to avoid NaN losses."* 4th place used BF16 throughout.

---

## Chapter 17. Overfitting, and the whole toolbox against it

### 17.1 The problem

With 773 wells and a 200-million-parameter model, the model can simply **memorise** the training wells. Training loss goes to zero; performance on new wells is terrible. This is **overfitting** — learning the noise instead of the signal.

The whole rest of this chapter, plus Chapters 18 and Part VI, is the arsenal against it.

### 17.2 Weight decay

Add `λ·||θ||²` to the loss, which pushes all weights toward zero unless the data really insists otherwise. Simpler functions generalise better. Typical values here: 1e-4 (3rd place's Last-PS NN), 0.01 (their Delta NN).

### 17.3 Dropout

During training, randomly zero out a fraction of the activations in a layer. The network cannot rely on any single pathway, so it learns redundant, distributed representations. At inference, dropout is off.

Note the spread of values in this competition: 3rd place used dropout 0.1 on their Last-PS NN but **0.5** on the Delta NN and SDF model. High dropout on the harder-to-generalise components.

### 17.4 Early stopping

Watch performance on held-out data; stop when it stops improving. 5th place: *"Fine-tuning on real data ran for 8 epochs, but the best epoch was almost always epoch 2."* With so little real data, fine-tuning has to be brief.

### 17.5 Data augmentation — the big one

Make more training data by transforming what you have in ways that preserve the answer. If you know that a well flipped left-to-right is still a valid well, you have doubled your dataset for free.

This competition was won on augmentation. The full catalogue used by the top five:

| Augmentation | What it does | Why it helps here | Used by |
|---|---|---|---|
| **GR affine** `GR' = a·GR + b` | rescale/shift the gamma ray | forces reliance on *shape*, not absolute level → immune to GR bias (H5) | 1st, 4th |
| **Z-shift / TVT resampling** | resample a plausible TVT path, keeping `TVT + Z` fixed | generates new physically valid paths (uses ★!) | 1st |
| **Fault-jump simulation** | inject sudden offsets | teaches the model rare catastrophes (H6) | 1st |
| **MD flip / reverse path** | traverse the well backwards | doubles data; physics is symmetric | 1st, 2nd, 4th, 5th |
| **Level / vertical flip** | flip the TVT axis (with label sign flip) | doubles data | 2nd, 4th |
| **MD stretch / time warp** | compress or stretch along MD | simulates different drilling rates & bed thicknesses | 1st, 4th |
| **Mixup** | train on `λ·x1 + (1−λ)·x2` with blended labels | smooths the decision surface | 4th |
| **Manifold mixup** | mixup applied to internal representations, not inputs | same, deeper in the net | 3rd (Delta NN, p=0.5, α=0.2) |
| **GR masking / dropout / SpecAug** | blank out spans of gamma ray | simulates real missing data (H5) | 1st, 2nd, 3rd, 4th |
| **Channel dropout** | zero entire input channels | stops over-reliance on any one feature | 4th |
| **Typewell swap** | pair a lateral with a different typewell | robustness | 4th |
| **Prior shift / slope noise / slope break** | corrupt the geometric baseline | teaches the model to fix a wrong prior | 4th |
| **Random known-fraction** | vary where PS is | robustness to prefix length | 4th |
| **Tail crop** | cut the end off | robustness to well length | 1st |
| **PF channel corruption** | rotate/shift particle-filter feature maps | prevents "shortcut learning" off the PF channels | 1st |

That last one is worth pausing on. 1st place fed a particle filter's output *as an input channel*. The risk is that the network learns to just copy that channel and ignore the gamma ray — a **shortcut**. Deliberately corrupting the channel during training breaks the shortcut and forces the network to keep using the raw data. That is a sophisticated, non-obvious move.

---

## Chapter 18. Trees, boosting, and the tabular models

Not everything here is a neural network. Two solutions used gradient-boosted trees.

**A decision tree** asks a series of yes/no questions about the features and lands you in a leaf with a prediction. Easy to fit, easy to overfit, not very accurate alone.

**Gradient boosting** builds hundreds or thousands of small trees *in sequence*, where each new tree is trained to predict the **residual error** left by all the previous trees combined. The sum of all trees is the prediction. **LightGBM** is the fast, standard implementation.

Boosted trees are outstanding on tabular data — rows of independent features. They are poor at problems with long-range sequential structure, which is exactly why 4th place's Alijs found that *"this competition wasn't very friendly to an LGB-style approach, as it worked at the row level and was missing the whole well-level picture."*

Where they *did* work here:

- **Residual correction.** Alijs ran a particle filter, then trained LightGBM on ~115 "self-diagnostic" features (particle spread, forward/backward disagreement, agreement with a second decoder) to predict *how wrong the particle filter was*, and subtracted that. Predicting your own errors is a powerful and underused pattern.
- **Quantile regression for the structural slope.** 4th place's James used LightGBM quantile regression to predict upper and lower confidence bounds on `dA/dM` (the structural slope) from `X, Y, azimuth, pitch`. **Quantile regression** means fitting the 10th percentile, or 90th percentile, of the target rather than the mean — done by using the *pinball loss*:

```
    L_τ(e)  =  τ·e         if e ≥ 0
               (τ−1)·e     if e < 0
```

which is asymmetric, so the minimiser is the τ-th quantile. Fit τ = 0.1 and τ = 0.9 and you have an 80% interval, with no distributional assumption. James then rendered that interval **as an image channel** so the downstream vision model could see how confident the slope estimate was.

- **TabPFN and RealMLP** also appear in James's stronger topography ensemble. **TabPFN** is a transformer pretrained on millions of synthetic tabular datasets that performs in-context learning: you feed it your whole (small) training table and it predicts without gradient training. **RealMLP** is a heavily-tuned MLP recipe competitive with GBMs on tabular data.

---

## Chapter 19. Validation — the chapter that decided this competition

### 19.1 The basic idea

You cannot judge a model by its training loss; it has seen those answers. Hold out data.

**k-fold cross-validation:** split the data into `k` parts. Train `k` models, each holding out one part. Every data point gets a prediction from a model that never saw it. Those are **OOF** (out-of-fold) predictions, and their score is your **CV**.

### 19.2 GroupKFold — and why a plain split is catastrophic here

Rows within a well are enormously correlated. Row 4,001 of well 17 looks almost exactly like row 4,000 of well 17. Split rows randomly, and nearly every validation row has a near-twin in training. Your CV score will be beautiful and completely fake. This is **leakage**.

**GroupKFold** splits by *group*, keeping all rows of a well together. 3rd place: *"Rows from the same well are strongly correlated, so a row-wise split causes severe leakage. We used five-fold GroupKFold with `well_id` as the group and always placed every row of one well in the same fold."* Their split gave 155/155/155/154/154 validation wells.

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for fold, (tr, va) in enumerate(gkf.split(X, y, groups=well_id)):
    ...   # no well_id appears in both tr and va
```

### 19.3 "Fold-safe": leakage is subtler than you think

3rd place's pipeline stacks models on models: physics models feed neural networks, which feed a gating network. Every one of those handoffs is a chance to leak. Their definition of **fold-safe** is worth memorising, because it is a complete checklist:

1. **Well-level split.** All rows of a well in one fold.
2. **Fold-local preprocessing.** Feature means and standard deviations computed on training folds only, then applied to validation. (Normalising with statistics from the whole dataset leaks.)
3. **OOF candidates.** When model B takes model A's predictions as input, the prediction for well W must come from an A-model that never trained on W.
4. **Synchronised folds.** Base models and gate use the *same* well assignment, so the gate never sees an in-fold base prediction.
5. **OOF-only blending.** Ensemble weights chosen on OOF, never on the public leaderboard.
6. **Fold averaging at test time.** All `k` fold models predict the test set; average them.

They spell out the consequence of getting #3 wrong: *"Without this two-stage OOF protocol, the gate could exploit in-fold base-model errors and produce an unrealistically optimistic Local CV."* Precisely — the gate would learn "trust model A, it's always right", which is only true on data A memorised.

### 19.4 Multiple split patterns

3rd place went further and ran **five different GroupKFold assignments** (they call them `default, split101, split102, split202, split303`), each reshuffling which wells land in which fold, giving 5 splits × 5 folds = 25 checkpoints per model family. They verified the assignments were nearly independent by computing pairwise Adjusted Rand Index between them (values from −0.0013 to 0.0038 — essentially zero overlap in structure).

Why: with only 773 wells, *which* wells landed together in a fold is itself a source of noise. Averaging over several assignments removes that noise. Measured payoff: going from 3 splits to 5 improved private LB from 5.903 to 5.836.

### 19.5 The fold-0 trap

4th place's Arunodhayan documented this beautifully:

> *"Fold-0 trap. Four configs won on fold 0 and lost on the 5-fold mean. Mixup improved fold 0 by −0.112 while folds 2 and 4 each degraded by +0.38."*

If you evaluate on one fold to save time, you will chase noise. They also measured the **noise floor** — the run-to-run standard deviation of the 5-fold mean was 0.103 for one recipe and 0.192 for another. Any "improvement" smaller than that is meaningless. Measuring your own noise floor before trusting a result is one of the most valuable habits in this entire book.

And on selection bias:

> *"Across 16 seeds, no seed beat the order-statistic prediction (best-of-8: predicted 6.6387, observed 6.6389)."*

Meaning: if you train 8 models and pick the best, its apparent advantage is exactly what pure chance predicts. Picking the best of N is not model improvement, it is a lottery.

### 19.6 🔬 The leave-largest-contribution-out test (2nd place)

This is the most statistically sophisticated idea in the five writeups, and it is fully general — use it in any competition with a squared-error metric.

**The problem.** Your new model scores 0.28 better on OOF. Is that real, or did it get lucky on a handful of wells?

**The test.** For each well `w`, compute the difference in summed squared error between the two models:

```
    g_w  =  SSE_treat(w)  −  SSE_base(w)          # negative = treatment is better
```

Now remove wells one at a time, in descending order of `|g_w|` — biggest contributors first — and recompute the total improvement each time. Define `k*` as the number of removals at which the improvement first vanishes.

- If `k*` is small, the entire gain rests on a few lucky wells. **Reject.**
- If `k*` is large, the gain is broad-based. **Accept.**

Their real example: one candidate looked 0.28 better, but removing just **8 of 773 wells** erased the gain → rejected. Another decayed gently and survived removing **52** wells (deliberately chosen as the size of the public LB) → adopted.

```python
import numpy as np

def leave_largest_out_curve(sse_base, sse_treat):
    """sse_* : arrays of per-well summed squared error. Returns improvement vs k."""
    g = sse_treat - sse_base                      # negative is good
    order = np.argsort(-np.abs(g))                # biggest |contribution| first
    total = g.sum()
    curve = [total]
    for idx in order:
        total -= g[idx]
        curve.append(total)
    return np.array(curve)                        # curve[k] = gain after removing k wells

curve  = leave_largest_out_curve(sse_base, sse_treat)
gone   = np.flatnonzero(curve >= 0)               # gain has disappeared
k_star = int(gone[0]) if gone.size else len(curve)   # never disappears -> maximally robust
# reject the candidate unless k_star comfortably exceeds the public-LB size
```


---

## Chapter 20. Neural networks, built up piece by piece

### 20.1 The neuron and the MLP

A single artificial neuron takes inputs `x`, multiplies each by a weight, sums, adds a bias, and passes the result through a non-linear function:

```
    a  =  σ( w·x + b )
```

`σ` is the **activation function** — historically sigmoid or tanh, nowadays **ReLU** (`max(0, x)`) or its smoother cousins **GELU** and **SiLU**. Without the non-linearity, stacking layers would be pointless: a composition of linear functions is just another linear function.

Stack neurons into **layers**, stack layers into a **multi-layer perceptron (MLP)**. An MLP with enough neurons can approximate any function — but it treats its inputs as an unstructured bag of numbers, which is wasteful when the inputs have structure.

### 20.2 Convolution: the right prior for signals and images

Gamma ray data has structure: nearby measurements are related, and a *pattern* (a spike, a ramp) means the same thing wherever it appears. A **convolutional layer** encodes both facts.

A convolution slides a small window (the **kernel**, e.g. 3×3 pixels or 7 samples) across the input, computing a weighted sum at each position — with **the same weights everywhere**.

Two consequences:

- **Parameter sharing.** A 3×3 kernel is 9 numbers regardless of image size.
- **Translation equivariance.** A feature detector that finds a gamma ray spike finds it anywhere along the well.

A layer has many kernels (**channels**), each learning a different pattern. Stack convolutions and the patterns compose: edges → textures → shapes → objects.

**Receptive field** = how much of the original input a single deep neuron can see. Each 3×3 layer widens it by 2. Ten layers → 21 samples. That is not enough to see a whole 10,000-ft well, which is why you need one of:

- **Pooling / striding** — downsample the feature map (e.g. average 2×2 blocks into one). Halves the resolution, doubles the effective receptive field, cheap. 1st place found that *"simple average pooling + interpolation performs better than learnable alternatives"* for down/upsampling.
- **Dilated convolution** — a kernel with gaps. A dilation-8 kernel of size 3 looks at positions `i−8, i, i+8`. Stack dilations 1, 2, 4, ..., 128 and your receptive field grows *exponentially* with depth while keeping full resolution. 3rd place's neural networks used exactly this: *"16 full-resolution dilated-TCN residual blocks. Kernel size 7; dilation cycle 1,2,4,...,128, repeated twice."*
- **Attention** — see 20.7.

```python
import torch.nn as nn
# a 1D dilated residual block, the workhorse of 3rd place's networks
class DilatedBlock(nn.Module):
    def __init__(self, c, k=7, d=1):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, k, padding=d*(k-1)//2, dilation=d)
        self.conv2 = nn.Conv1d(c, c, k, padding=d*(k-1)//2, dilation=d)
        self.norm  = nn.GroupNorm(8, c)
        self.act   = nn.GELU()
    def forward(self, x):
        h = self.act(self.norm(self.conv1(x)))
        h = self.conv2(h)
        return x + h                      # residual connection
```

### 20.3 Normalisation layers

Deep networks train badly if activations drift to huge or tiny values. Normalisation layers rescale activations to roughly zero mean, unit variance.

- **BatchNorm** normalises using statistics of the current *batch*. Very effective, but couples samples in a batch, and misbehaves with small batches.
- **LayerNorm** normalises each sample independently across its features. Standard in transformers and in ConvNeXt.
- **GroupNorm** normalises within groups of channels. A middle ground.

1st place found something you would not guess: **replacing ConvNeXt's LayerNorm with BatchNorm consistently worked better** in this task — at the cost of needing BF16 to avoid NaNs. Never assume the default is optimal.

### 20.4 Residual connections

`output = x + F(x)` instead of `output = F(x)`. The block only has to learn a *correction*, and gradients flow straight through the `+` to earlier layers. This one line is what made networks deeper than ~20 layers trainable, and it is why **ResNet** was such a landmark. Every backbone named in this competition uses residual connections.

### 20.5 The backbone zoo, in one table

A **backbone** is a pretrained feature extractor — usually trained on ImageNet (a million labelled photographs) — that you reuse. The library everyone uses is `timm`.

| Backbone | Family | One-line description | Used by |
|---|---|---|---|
| **ResNet-34** | CNN | The classic residual CNN. Small, fast, reliable. | 4th (Lightsource, Arunodhayan) |
| **EfficientNet-B0 / V2-S** | CNN | Compound-scaled CNN designed for accuracy per FLOP. | 2nd (b0), 4th, 5th (`effnetv2_rw_s`) |
| **HRNet-W18** | CNN | Keeps a *high-resolution* branch throughout instead of downsampling then upsampling. Good for dense prediction. | 5th |
| **ConvNeXt (S / V2-L)** | CNN | A ResNet modernised with transformer-era design choices (large kernels, LayerNorm, GELU, inverted bottlenecks). The strongest pure CNN family. | **1st** (`convnext_small.in12k_ft_in1k_384`), 4th (`convnextv2_large.fcmae_ft_in22k_in1k_384`), 4th (Lightsource) |
| **Swin-Tiny** | Transformer | Vision transformer with shifted local windows — attention, but at CNN-like cost. | 4th (Lightsource) |
| **MaxViT (T / B)** | Hybrid | Alternates local window attention and sparse *grid* attention, plus convolutions. | 5th (tiny, then base) |
| **MambaVision** | SSM hybrid | State-space model backbone; linear-time alternative to attention. | 4th (James — promising at small scale, no time to scale up) |

**The headline finding, from 1st place:** *"In my experiments, CNN-based backbones consistently performed better, and only ConvNeXt contributed to the final solution."* 4th place's James agreed: ConvNeXt beat ViTs and EfficientNetV2 in early experiments. With ~700 real training wells, transformers' weaker inductive bias hurts.

Note the model-name suffixes; they are informative. `convnextv2_large.fcmae_ft_in22k_in1k_384` reads as: ConvNeXt V2, Large size, pretrained with **FCMAE** (a masked-autoencoder self-supervised method), then fine-tuned on **ImageNet-22k**, then on **ImageNet-1k**, at **384×384** resolution.

### 20.6 U-Net and the encoder–decoder shape

For **dense prediction** — an output for every pixel/position, not one number per image — the standard architecture is the **U-Net**.

```
   input ──► [enc1] ──────────────────────────────► [dec1] ──► output
                │  ↓                             ↑  │
                └─►[enc2] ──────────────► [dec2] ─┘
                       │  ↓            ↑  │
                       └─►[enc3] ─► [dec3]─┘
                              │  ↓  ↑
                              └►[bottleneck]
```

The **encoder** downsamples repeatedly, building abstract, wide-context features. The **decoder** upsamples back to full resolution. **Skip connections** carry high-resolution detail from each encoder level straight across to the matching decoder level, so fine detail is not lost in the bottleneck.

You need both here: wide context to know roughly where in the barcode you are, fine detail to pin the boundary to the nearest half-foot.

**Feature Pyramid Network (FPN)** is a lighter relative. Instead of a full mirrored decoder, project every encoder stage to a common channel width with 1×1 convolutions, resize them all to one resolution, and add them. 2nd place used exactly this: *"the outputs of every efficientnet-b0 stage are fused by an FPN-style decoder (each stage projected to a common width by 1×1 conv, interpolated to a single resolution, and summed)."*

4th place's James found the multi-scale idea mattered even more than usual: *"I found it beneficial to use features from multiple spatial scales in the vision backbone, not just pooled features from the final layer. However, this workload seemed to benefit from an even more extreme form of that: using all 4 spatial scales instead of only 2."*

He also found the **head** should be shallow: *"it works best with relatively 'shallow' heads with a small number of channels in their internal representations (forcing the backbone to do most of the work)."* With little real data, put your capacity in the pretrained part and keep the freshly-initialised part small.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class FPNHead(nn.Module):
    """Project every backbone stage to a common width, resize, sum. 2nd place's decoder."""
    def __init__(self, in_chs, width=64):
        super().__init__()
        self.lat = nn.ModuleList(nn.Conv2d(c, width, 1) for c in in_chs)
    def forward(self, feats, out_hw):
        out = 0
        for f, l in zip(feats, self.lat):
            out = out + F.interpolate(l(f), size=out_hw, mode='bilinear', align_corners=False)
        return out
```

### 20.7 Attention and transformers, briefly

A convolution looks at a fixed neighbourhood. **Attention** lets every position look at every other position and decide, dynamically, which ones matter.

For each position, produce a **query** `q`, a **key** `k`, and a **value** `v`. The output at position `i` is a weighted average of all values, where the weight on position `j` is how well `q_i` matches `k_j`:

```
    Attention(Q,K,V) = softmax( Q Kᵀ / √d ) V
```

**Self-attention** is when Q, K, V all come from the same sequence. **Cross-attention** is when Q comes from one sequence and K, V from another — which is exactly how 3rd place let their lateral-well encoder consult the typewell: *"Downsampled typewell cross-attention after block 8."* The lateral asks "what does the typewell look like around here?" and the typewell answers.

The cost is `O(n²)` in sequence length, which is why variants exist:

- **Swin** — attention inside local windows, with the windows shifted between layers so information still spreads.
- **MaxViT** — alternates local window attention with *sparse grid* attention (attend to every 8th position), getting global reach cheaply. 3rd place used a 1D analogue: *"MaxViT1D local-window/sparse-grid attention after blocks 4, 8, 12, and 16."*
- **Squeezeformer** — a Conformer variant (convolution + attention blocks interleaved) from speech recognition, good for long 1D sequences. 4th place's Lightsource used it for their 1D model.
- **Mamba / state-space models** — replace attention with a learned linear recurrence that runs in linear time.

### 20.8 Recurrent networks: LSTM and BiLSTM

An RNN processes a sequence step by step, carrying a **hidden state** forward. An **LSTM** adds gates (input, forget, output) that control what enters, persists in, and leaves that state, which fixes the vanishing-gradient problem of plain RNNs.

A **BiLSTM** runs two LSTMs, one forward and one backward, and concatenates their states, so every position sees the whole sequence in both directions.

Largely superseded by transformers for language, but still excellent for small data and modest sequence lengths — which describes this competition exactly. 3rd place used a gated BiLSTM in their typewell encoder, a gated LSTM output head, and a **BiLSTM SoftMax Gate** as their final ensemble mechanism (which beat their CNN gate: 5.3329 vs 5.4255 OOF).

### 20.9 Transfer learning: pretrain, then fine-tune

Train a model on a large dataset, then continue training it on your small one. The early layers have already learned generic feature detectors — edges, textures, gradients — that transfer.

Standard practice everywhere. In this competition it appears in *two* forms, and it is important to distinguish them:

1. **ImageNet pretraining.** Start from `timm` weights trained on photographs. Free, and it works even though gamma ray canvases look nothing like photographs.
2. **Synthetic-data pretraining.** Generate hundreds of thousands of *fake wells* from a physical model, pretrain on those, then fine-tune on the 773 real ones. This is Part VI, and it is arguably what separated the top five from everyone else.

4th place's James stacked all three stages: ImageNet/FCMAE weights → 500k synthetic "V0" wells → 37k synthetic "V3" wells → real wells. He measured the payoff of the synthetic stage in an early experiment: **8.2 → 6.7 CV RMSE**, about 1.5 ft.

### 20.10 🔬 Anchors: the object-detection idea, borrowed

2nd place's architecture is named **AnchorCNN** after an idea from object detection (YOLO, CenterNet).

The problem those detectors solve: an image contains an unknown number of objects at unknown places. The solution: lay a grid over the image, and at every grid cell, predict "is there an object centred here, and if so, what shape and class?" Each grid cell is an **anchor** — a fixed position that owns a prediction.

2nd place's insight was to map that structure onto geosteering. Their grid cell `(i, j)` means "**MD column j, TVT level i**" — that is, a *hypothesis about where the well currently is*. From the feature vector at that cell, the model predicts: *"if the well were at level i at column j, how far up or down would it move next?"*

So the CNN produces, in one forward pass, an answer for **every possible position hypothesis simultaneously**. That is what makes the conditional distribution `P(ΔTVT | TVT)` computable everywhere, which is what makes exact dynamic-programming decoding possible (Chapter 26).

---

## Chapter 21. Ensembling: the last 0.5 RMSE

An **ensemble** is several models whose predictions are combined. Ensembles beat their members almost always, because different models make **different mistakes**, and averaging cancels independent errors.

The maths: if `M` models each have error variance `σ²` and their errors are uncorrelated, the average has variance `σ²/M`. Correlated errors don't cancel — so **diversity is the whole game**. This is why 1st place kept particle-filter models in their ensemble *even though those models no longer improved the single best model*: "I kept them in my final ensemble for diversity."

### 21.1 Sources of diversity, as used here

- different random **seeds** (1st place: 3 seeds × 5 folds)
- different **folds** (everyone)
- different **fold assignments** (3rd place: 5 split patterns)
- different **backbones** (4th place: ResNet-34 + ConvNeXt-Tiny + Swin-Tiny + Squeezeformer; 5th place: MaxViT + EfficientNetV2 + HRNet)
- different **input features** (1st place: with/without particle filter channels, with/without XY-neighbour channels)
- different **augmentation recipes** (4th place's four fine-tuning "trials" differ mainly in augmentation)
- different **loss functions and thresholds** (4th place: Huber δ = 9.48 / 0.549 / 3.013 / MSE)
- **fundamentally different model classes** (3rd place: HMM + particle filter + two neural nets + a correlation model)

### 21.2 How to combine

**Simple average.** Hard to beat. 4th place's Arunodhayan measured *"equal-weight ensembling: −0.43"* and, separately, that **fitted weights lost to equal weights** under proper nested validation:

| pool | in-sample | nested | equal |
|---|---|---|---|
| 15 members | 6.385 | 6.578 | **6.456** |
| 12 members | 6.188 | 6.323 | **6.244** |

Read that table carefully. In-sample, fitted weights look best (6.385). Under honest nested cross-validation, they are *worse* than just averaging (6.578 vs 6.456). Fitting weights is itself a fitting procedure, and it overfits.

**Weighted average.** Weights chosen on OOF. 1st place hand-set weights per model and, importantly, used **two different weight vectors**: one for normal wells and one for the ~10% of wells with unreliable XY-neighbour information.

**Ridge / NNLS stacking.** Fit a regression from member predictions to the truth. 4th place's team used ridge regression on OOF to set the three top-level weights (0.644 / 0.327 / 0.029); Lightsource used **NNLS** (non-negative least squares) inside his own sub-ensemble. NNLS forces weights ≥ 0, which is a useful regulariser — negative weights are usually a sign of overfitting.

**Learned gating.** 3rd place's contribution: a small network that outputs *different weights at every row*, via softmax over candidates. Chapter 41.7 covers it fully.

### 21.3 Test-Time Augmentation (TTA)

Apply your augmentations at *inference* time, predict on each version, undo the transform, average. Same variance-reduction logic, no extra training.

The TTAs used here are unusually clever and worth studying:

- **MD-phase TTA (2nd place, 8 views).** Their CNN bins MD into 32-ft columns, so *where you cut the column boundaries* changes the input (aliasing). They ran inference at 8 sub-column offsets — {0, 8, 16, 24, 32, 40, 48, 56} ft — and averaged. This recovers sub-column resolution that the coarse grid threw away. Worth 5.624 → 5.417 OOF. (Asked whether an AI suggested it, the author replied that the AI proposed "MD shift" but *he* specified that the shift should be sub-column and undone afterwards, and the number 8 came from a sweep.)
- **Physically-correct vertical flip (4th place, +0.13).** Not a naive image flip: they simulate what the data would look like if the entire rock column were upside down *and* the drill were heading the opposite vertical direction, so that the topography channel still makes physical sense. Then invert the predictions back and average.
- **Quarter re-anchoring (4th place, +0.18).** Predict the whole well. Then pretend the prediction at 25% of the way through was a *known* value, re-render the input from there, and predict the last 75% again. Average. This resets accumulated drift. They tried adding a 50% re-anchor (helped CV, hurt private LB) and a 75% one (hurt both), and stopped at one.
- **Multiple grid resolutions (2nd place).** Ensemble a 32-ft/column model with a 16-ft/column model.

---

## Chapter 22. A vocabulary checkpoint

Before Part IV, make sure these are solid. If any is fuzzy, reread the chapter in brackets.

`loss` [15] · `gradient descent` [16] · `AdamW` [16.2] · `warmup` [16.3] · `EMA` [16.4] · `BF16` [16.5] · `overfitting` [17] · `dropout` [17.3] · `augmentation` [17.5] · `mixup` [17.5] · `LightGBM` [18] · `quantile regression` [18] · `GroupKFold` [19.2] · `OOF` [19.1] · `leakage` [19.2] · `fold-safe` [19.3] · `convolution` [20.2] · `dilation` [20.2] · `receptive field` [20.2] · `BatchNorm/LayerNorm` [20.3] · `residual connection` [20.4] · `backbone` [20.5] · `U-Net` [20.6] · `FPN` [20.6] · `attention` [20.7] · `cross-attention` [20.7] · `BiLSTM` [20.8] · `transfer learning` [20.9] · `anchor` [20.10] · `ensemble` [21] · `TTA` [21.3] · `NNLS` [21.2]


---

# PART IV — GUESSING WHERE YOU ARE
## Probability, hidden states, HMMs, particle filters, and dynamic programming

Half of this competition is neural networks. The other half is a family of methods from robotics and signal processing that answer one question: *"given a stream of noisy measurements, where am I?"* That is literally the problem a self-driving car solves, and literally the problem a drill bit solves. Two of the top three solutions have one of these at their core.

---

## Chapter 23. Probability, in the amount you need

**Random variable.** A quantity whose value is uncertain. `TVT at row 5000` is one.

**Distribution.** The full description of how likely each value is. For a continuous quantity, a **probability density** `p(x)`; the area under it is 1.

**The Gaussian (normal) distribution.**

```
    p(x) = (1 / (σ√(2π))) · exp( −(x − μ)² / (2σ²) )
```

Bell-shaped, defined by mean `μ` and standard deviation `σ`. Ubiquitous because sums of many small independent effects tend toward it.

**Heavy-tailed distributions.** The Gaussian says "5σ events essentially never happen". Real data disagrees. The **Student-t** distribution looks like a Gaussian in the middle but has fatter tails; with 1 degree of freedom it is the **Cauchy**, which has tails so fat it has no mean. Using it as a likelihood means a single bizarre gamma ray reading barely moves your beliefs — exactly the robustness 3rd place wanted from their HMM emission model.

**Likelihood.** Given a candidate explanation `h` and an observation `o`, the likelihood `p(o | h)` is "how probable was this observation, if `h` were true?" Note the direction: it is a function of the *hypothesis*, evaluated at fixed data.

**Bayes' rule.** How to update a belief when evidence arrives:

```
                       p(observation | hypothesis) · p(hypothesis)
    p(hypothesis|obs) = ───────────────────────────────────────────
                                 p(observation)
    
       POSTERIOR      ∝        LIKELIHOOD         ×        PRIOR
```

In words: **new belief ∝ how well the hypothesis explains the data × how plausible the hypothesis was beforehand.**

For us:
- **hypothesis** = "the bit is at TVT = 7.5 ft right now";
- **prior** = what we believed a foot ago, propagated forward by the physics of drilling;
- **likelihood** = how close the typewell's gamma ray at TVT = 7.5 is to the gamma ray we just measured;
- **posterior** = the updated belief.

Run that loop once per foot for 8,000 feet, and you have solved the problem. That loop is called **filtering**, and the rest of Part IV is three different ways to implement it.

---

## Chapter 24. State-space models: the shape of the problem

A **state-space model** has two pieces.

**The state** `s_t` — everything you need to know about the system at time `t` to predict its future. It is hidden; you never observe it directly.

**Two model components:**

```
    Transition:   p(s_t | s_{t−1})       "how the state evolves"
    Emission:     p(o_t | s_t)           "what you'd measure if the state were s_t"
```

For geosteering, a rich state (this is essentially 3rd place's HMM state) is:

| Component | Meaning |
|---|---|
| `TVT` | stratigraphic position — the thing we actually want |
| `rate` = `d(TVT + Z)/dMD` | the structural slope, i.e. `dz_layer/dMD` from (★) |
| `bias` | the slowly drifting GR offset between this well and the reference |
| `reference` | *which* reference curve is right (typewell? siblings?) |

**Transition:** TVT moves according to the rate and the known change in Z; the rate itself drifts slowly (and occasionally jumps, at a fault); the bias drifts slowly.

**Emission:** observed GR ≈ reference_GR(TVT) + bias + noise.

Note that including `rate` in the state is what gives the model *momentum*. Without it, the filter has no memory of which way the layers were dipping and will wander. Including it means "the layers have been rising at 0.002 ft/ft for the last 2,000 feet, so they probably still are." This is the state-space encoding of equation (★).

Three algorithms solve state-space models, and they differ in how they represent the belief:

| Method | Belief representation | Works when |
|---|---|---|
| **Kalman filter** | one Gaussian (mean + covariance) | everything is linear and Gaussian |
| **HMM / grid filter** | a probability for each cell of a discretised grid | the state space is small enough to enumerate |
| **Particle filter** | a cloud of weighted samples | anything, but you need enough samples |

Kalman is out — our belief is *multimodal* (H1) and a single Gaussian cannot represent "either +10 or −10". So: HMM or particle filter. This competition used both, and 3rd place used both *together*.

---

## Chapter 25. Hidden Markov Models

### 25.1 The setup

Discretise the state into `S` possible values. (For us: chop TVT into bins — 3rd place used bins as fine as 0.0625 ft in one HMM variant.) Then you need:

- `π(s)` — the initial distribution over states;
- `A(s' → s)` — the transition matrix, the probability of moving from `s'` to `s` in one step;
- `B(s, o)` — the emission probability of observing `o` in state `s`.

"Markov" means the future depends on the present only, not on the whole past. That is why the state must contain everything relevant (like the rate).

### 25.2 Forward: filtering

Let `α_t(s)` = probability of being in state `s` at time `t`, given observations `1..t`. Then:

```
    α_t(s)  ∝  B(s, o_t) · Σ_{s'} A(s' → s) · α_{t−1}(s')
                └────────┘   └──────────────────────────┘
                 likelihood        predicted prior
```

Predict forward, then reweight by how well each state explains the new observation, then renormalise. That is Bayes' rule, once per step.

### 25.3 Backward and smoothing: using the future

The forward pass at time `t` only knows the past. But once you have seen the *whole* well, information from row 8,000 should sharpen your belief about row 3,000. That is **smoothing**, and it uses a symmetric backward pass:

```
    β_t(s)  =  Σ_{s'} A(s → s') · B(s', o_{t+1}) · β_{t+1}(s')

    γ_t(s)  ∝  α_t(s) · β_t(s)        ← the smoothed posterior
```

`γ_t` is the probability of being in state `s` at time `t` **given every observation in the well**. Since we are predicting offline, not in real time, there is no reason not to use it. 3rd place: *"Forward-backward smoothing then produced a posterior mean conditioned on the complete sequence."*

### 25.4 Decoding: three choices, and why one wins

Given `γ_t(s)`, how do you produce a single number?

| Decoder | Rule | Optimal for |
|---|---|---|
| **Argmax / marginal MAP** | `argmax_s γ_t(s)` | 0-1 loss per step |
| **Viterbi** | the single most probable *whole path* | 0-1 loss on the entire path |
| **Posterior mean** | `Σ_s γ_t(s) · TVT(s)` | **squared error** ← our metric |

Viterbi is famous, elegant, and gives you a physically consistent path. It is also **wrong for this competition**. It commits to one hypothesis; when there are two equally plausible modes it picks one and eats the full squared error if it picked wrong (Chapter 10.3).

2nd place tested this directly: *"Viterbi (maximum-likelihood path) and other mode-tracking decoders were consistently worse than the expectation."*

Learn the rule: **the decoder must match the metric.**

### 25.5 A minimal HMM you can run

```python
import numpy as np

def hmm_forward_backward(log_emis, log_trans, log_init):
    """
    log_emis : (T, S)  log p(observation_t | state s)
    log_trans: (S, S)  log p(s_t = j | s_{t-1} = i)   [rows = from]
    log_init : (S,)    log p(s_0)
    returns gamma: (T, S) smoothed posterior
    """
    T, S = log_emis.shape

    def lse(a, axis):                       # numerically safe log-sum-exp
        m = a.max(axis=axis, keepdims=True)
        return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)

    log_a = np.zeros((T, S)); log_b = np.zeros((T, S))
    log_a[0] = log_init + log_emis[0]
    for t in range(1, T):
        log_a[t] = lse(log_a[t-1][:, None] + log_trans, axis=0) + log_emis[t]
    for t in range(T - 2, -1, -1):
        log_b[t] = lse(log_trans + (log_emis[t+1] + log_b[t+1])[None, :], axis=1)

    log_g = log_a + log_b
    log_g -= lse(log_g, axis=1)[:, None]
    return np.exp(log_g)

# --- geosteering flavour -------------------------------------------------
# states = TVT bins;  transition centred on the physics prediction from (★)
tvt_bins = np.arange(-100, 100, 0.5)
S = len(tvt_bins)

def build_transition(dz, rate, sigma=0.6, jump_p=1e-4):
    """Expected step is (rate*dMD - dz); allow rare large jumps for faults."""
    step = rate - dz                                   # per (★), in ft
    d = tvt_bins[None, :] - tvt_bins[:, None] - step   # deviation from expectation
    A = np.exp(-0.5 * (d / sigma) ** 2)
    A += jump_p                                        # heavy tail: fault jumps
    return np.log(A / A.sum(axis=1, keepdims=True))

def build_emission(gr_obs, typewell_gr_at_bins, bias=0.0, scale=8.0):
    """Student-t(df=1) = Cauchy: robust to outlier GR readings (3rd place's choice)."""
    r = (gr_obs - (typewell_gr_at_bins + bias)) / scale
    return -np.log1p(r ** 2)                           # log Cauchy kernel

# prediction = posterior MEAN, not argmax
# tvt_hat = (gamma * tvt_bins[None, :]).sum(axis=1)
```

Note the two design choices baked into that snippet, both taken from the real solutions: a **heavy-tailed transition** (`+ jump_p`) so faults are possible instead of impossible, and a **heavy-tailed emission** (Cauchy) so one weird gamma ray reading does not derail the filter.

### 25.6 What 3rd place actually built

Not one HMM but a **three-family mixture**, because the families make different mistakes:

| Family | Weight | Character |
|---|---|---|
| Base HMM (Exp417) | 0.50 | self-prefix reference, two reference mixtures, explicit bias state, exact forward-backward |
| Local-DTW HMM | 0.20 | local gamma ray pattern matching with a small allowed stretch |
| Fine-bin HMM | 0.30 | higher-resolution sibling reference at 0.0625 ft bins |

Their comment on the second one is the ensembling lesson of Chapter 21 in miniature: *"Although Local-DTW was not the strongest standalone model, its lower error correlation with the base HMM made it valuable in the ensemble."* The mixture improved OOF from 6.0492 to 5.9703.

Other details worth stealing:

- **Reference design mattered more than architecture.** Their #1 listed private-LB insight: use sibling-lateral GR (from other wells sharing the typewell), aggregate it in fine TVT bins, mix in the target well's own known-prefix GR as a self-reference, and **keep the reference identity as part of the hidden state instead of averaging references together**. Averaging two references blurs both; letting the filter decide which one to trust preserves both.
- **Warm-starting from the prefix.** Estimate the initial formation rate from the last **256** known rows (not 30 — that change alone was worth 5.854 → 5.812 on private); prepend the last 128 prefix rows to the HMM sequence; clamp their TVT states to the known values. So the filter enters the unknown region already locked onto the right dip and the right alignment.
- **Physical clamping of the output.** Project the predicted path using the 1st–99th percentile of formation rate estimated from the prefix; cap the absolute rate at 0.25; cap the *integrated* TVT correction at 10 ft. Note it is the integrated correction that is capped, not the per-row one — that is the right way to prevent slow drift without forbidding legitimate local movement.

---

## Chapter 26. Particle filters

### 26.1 The idea

An HMM enumerates the whole state space. That is fine for one dimension, but our state is really four-dimensional (TVT × rate × bias × reference family) and a full grid over four dimensions is enormous.

A **particle filter** (a.k.a. Sequential Monte Carlo) represents the belief as a **cloud of samples**. Each **particle** is one complete guess at the state — "TVT = 7.3, rate = 0.0021, bias = −4.1, using the sibling reference" — carrying a weight.

The loop, once per row:

1. **Predict.** Move every particle forward through the transition model, with random noise. Each particle now guesses where it went.
2. **Weight.** Multiply each particle's weight by the likelihood of the observed gamma ray under that particle's state. Particles that explain the data get heavier.
3. **Normalise.** Weights sum to 1.
4. **Resample (sometimes).** Draw a new set of particles from the old ones with probability proportional to weight. Good particles get duplicated, bad ones die.

The particle cloud *is* the posterior. It handles multimodality naturally: if there are three plausible layers, particles cluster into three groups.

### 26.2 Degeneracy, ESS, and resampling

Without step 4, after a few hundred steps one particle carries essentially all the weight and the rest are dead weight. This is **degeneracy**.

You measure it with the **Effective Sample Size**:

```
    ESS  =  1 / Σ_i w_i²
```

If all `N` weights are equal, ESS = N. If one particle has all the weight, ESS = 1. Standard practice: resample when ESS drops below some threshold (e.g. N/2). 3rd place: *"we applied systematic resampling when the Effective Sample Size fell below its threshold."*

**Systematic resampling** is a low-variance resampling scheme: instead of drawing `N` independent uniforms, draw one uniform and take `N` equally-spaced points from it along the cumulative weight axis. Cheaper and less noisy than multinomial resampling.

### 26.3 Smoothing: FFBSi and genealogy

Resampling destroys history. If you just record the current particle cloud at each step, the early part of your trajectory collapses to a single ancestor. Two fixes appear in the writeups:

- **Genealogy tracing** (3rd place): remember each particle's parent, then walk backwards from the surviving particles at the end to recover their full ancestral paths. Cheap. *"Seed-level paths were weighted using full-sequence likelihood and smoothed by tracing particle genealogy backward."*
- **FFBSi** — Forward Filtering Backward Simulation (1st place). A proper particle smoother: run the filter forward, then sample backwards, at each step choosing an ancestor with probability proportional to `w_i × p(s_{t+1} | s_t^i)`. More expensive, statistically much better, and it does not suffer path degeneracy.

### 26.4 A runnable particle filter for this problem

```python
import numpy as np
rng = np.random.default_rng(0)

def particle_filter(gr_obs, dz, tw_gr_fn, N=500,
                    rate0=0.0, rate_noise=1e-3, pos_noise=5e-3,
                    rate_momentum=0.9995, jump_p=2e-4, jump_scale=8.0,
                    gr_scale=8.0, ess_frac=0.5):
    """
    gr_obs : (T,)   observed gamma ray along MD
    dz     : (T,)   known change in Z per row
    tw_gr_fn(tvt) -> expected GR from the reference curve
    """
    T = len(gr_obs)
    tvt  = np.zeros(N)                     # relative to the anchor at PS
    rate = np.full(N, rate0)
    bias = rng.normal(0, 3.0, N)           # unknown GR offset per particle
    w    = np.full(N, 1.0 / N)
    out  = np.zeros(T)
    parents = np.zeros((T, N), dtype=np.int32)
    idx = np.arange(N)

    for t in range(T):
        # ---- 1. PREDICT: equation (★) plus noise, plus rare fault jumps
        rate = rate_momentum * rate + rng.normal(0, rate_noise, N)
        tvt  = tvt + rate - dz[t] + rng.normal(0, pos_noise, N)
        jump = rng.random(N) < jump_p
        tvt[jump] += rng.normal(0, jump_scale, jump.sum())
        bias = bias + rng.normal(0, 0.02, N)

        # ---- 2. WEIGHT: Cauchy likelihood on the GR mismatch
        if not np.isnan(gr_obs[t]):
            r = (gr_obs[t] - (tw_gr_fn(tvt) + bias)) / gr_scale
            w = w * (1.0 / (1.0 + r ** 2))
            s = w.sum()
            w = np.full(N, 1.0/N) if s <= 0 else w / s

        out[t] = float((w * tvt).sum())                 # EXPECTATION, not argmax

        # ---- 3. RESAMPLE when the cloud has degenerated
        ess = 1.0 / np.sum(w ** 2)
        if ess < ess_frac * N:
            pos = (rng.random() + np.arange(N)) / N     # systematic resampling
            pick = np.searchsorted(np.cumsum(w), pos)
            tvt, rate, bias, idx = tvt[pick], rate[pick], bias[pick], idx[pick]
            w = np.full(N, 1.0 / N)
        parents[t] = idx
    return out, parents
```

Three things in that code are lifted straight from the winners' notes:

- `jump_p` — **1st place**: *"Allowing low-probability large jumps"* was one of their listed major improvements over the public baseline. A filter that forbids jumps can never recover from a fault.
- `bias` as a per-particle state — **3rd place** carries GR bias explicitly.
- the **expectation** output rather than the argmax — Chapter 10.3, again.

### 26.5 What the top teams did beyond the basics

**1st place's improvements over the public baseline** (their standalone particle filter reached ~7.4 CV RMSE):
- allowing low-probability large jumps;
- **calibrated typewell GR** — adjusting the typewell curve using the visible region of the lateral before matching against it;
- **FFBSi** smoothing;
- **blending diverse configuration profiles** — running the filter several times with different settings and averaging;
- **updating particles in bins of 64 samples instead of at raw 1-ft resolution** — this both speeds things up and smooths the likelihood, which reduces the chance of latching onto a spurious single-sample match.

**3rd place's settings**: 500 particles × 240 seeds × 3 seed bases; rate momentum 0.9995; rate noise 0.001; position noise 0.005; two observation models blended 0.6/0.4 (`f015`, in which 15% of particles belong to a "hard" family whose likelihood is progressively strengthened near the start, and `soft`, in which all particles allow GR bias).

Note "240 seeds": they run the whole filter 240 times and combine. A particle filter is stochastic; a single run is one sample from a distribution over answers.

**Most importantly, 3rd place used the particle filter as a *feature extractor*, not just a predictor.** They fed these into their neural networks:

- weighted mean; spread; skewness; IQR;
- quantiles q10, q25, q50, q75, q90;
- previous-row and next-row differences of all of the above.

That is the particle filter handing the neural net a full **uncertainty profile** at every row: "here is my best guess, here is how spread out my hypotheses are, here is whether they're lopsided." A neural network given that can learn *when to trust the physics and when to override it*. This is one of the most transferable ideas in the entire competition.

**1st place** did the same thing in 2D: they rendered the particle cloud as a **2D probability heatmap** (TVT × MD) and fed it as an image channel — along with `|typewell_TVT − PF_TVT|`, the distance from each candidate to the filter's estimate.

**4th place's Alijs** ran the filter **forward and backward** and used the disagreement between the two directions as a feature for a LightGBM error-corrector. A forward filter and a backward filter that agree are probably right; where they diverge, something is wrong.

---

## Chapter 27. Dynamic programming and exact expectation decoding

### 27.1 The setup (2nd place's decoder)

Suppose a neural network has given you, for every MD column `k` and every TVT level `s`, a **conditional move distribution**:

```
    P( ΔTVT = δ | you are at level s, column k )
```

2nd place's AnchorCNN outputs exactly this: 21 discrete move classes, `{0, ±2, ±4, …, ±20} ft` per column, at every one of 128 levels × 336 columns.

Now, how do you get a path out of it?

**Option A: Rollout.** Start at the known PS level, sample a move, move, sample again... Repeat 1,024 times to get a bundle of paths. Average them. This works, but it is noisy, and you need a lot of samples.

**Option B: Dynamic programming.** Compute the exact answer, with no sampling at all.

Let `p_k(s)` = the probability that the path is at level `s` at column `k`. Start with all the mass on the known PS level. Then propagate:

```
    p_{k+1}(s)  =  Σ_{s'}  p_k(s') · P( ΔTVT = s − s' | anchor(s', k) )        ...(5)

    TVT_hat(k)  =  Σ_s  p_k(s) · TVT(s)                                        ...(6)
```

Equation (5) says: the probability of being at level `s` next column is the sum, over all levels you could have been at, of (probability you were there) × (probability of the move that gets you here). Equation (6) is the expectation decode.

That is it. Two lines, no sampling, exactly correct.

```python
import numpy as np

def dp_expectation(move_logprob, ps_level, moves, levels):
    """
    move_logprob : (K, S, M)  log P(move m | level s, column k)
    moves        : (M,)       move sizes, in LEVEL units (e.g. [-10..+10])
    levels       : (S,)       TVT value of each level bin
    """
    K, S, M = move_logprob.shape
    p = np.zeros(S); p[ps_level] = 1.0
    path = np.zeros(K)
    for k in range(K):
        path[k] = float((p * levels).sum())          # eq (6): expectation
        P = np.exp(move_logprob[k])                  # (S, M)
        nxt = np.zeros(S)                            # eq (5): propagate
        for m_i, dm in enumerate(moves):
            src = np.arange(S)
            dst = src + int(dm)
            ok = (dst >= 0) & (dst < S)
            np.add.at(nxt, dst[ok], p[src[ok]] * P[src[ok], m_i])
        s = nxt.sum()
        p = nxt / s if s > 0 else p
    return path
```

### 27.2 Why DP beat the alternatives here

2nd place's original plan was Option A plus a **reranker**: generate 1,024 rollout paths and train a second model to pick the good ones. They report that an oracle — one that cheats and picks the best path in hindsight — beats their deployed score by several feet. So the good paths *are* in the bundle. But:

> *"I built a reranker (a model that picks good paths from the bundle) and tried reinforcement learning (GRPO), but both selectors overfit the evaluation data easily and never consistently beat the DP expectation. In the end I abandoned the 'select from a bundle' design and committed to DP marginalisation."*

This is a recurring theme. **The oracle gap is real but unreachable**, because the selector needed to close it has to be learned from the same tiny dataset, and it overfits. 3rd place ran into exactly the same wall with their SoftMax gate: an oracle that picks the best candidate per row would score far better than their gate does, and they explicitly note the ceiling.

### 27.3 The relationship between all these decoders

Everything in Part IV computes a *distribution over paths* and then collapses it. The collapse is where the money is:

| Method | Distribution | Collapse |
|---|---|---|
| HMM | grid posterior `γ_t(s)` | posterior mean |
| Particle filter | weighted particle cloud | weighted mean |
| AnchorCNN + DP | propagated marginal `p_k(s)` | expectation, eq (6) |
| 2D U-Net + softmax | per-column probability over rows | expectation over rows |
| MTP head | M candidate paths + probabilities | probability-weighted average |

Five different machineries. **One collapse rule: take the expectation.** That is the unifying idea of the entire competition.

---

## Chapter 28. 🔬 Dynamic Time Warping, in one page

Two solutions mention DTW, so here it is.

You have two sequences that contain the same features but stretched differently — the typewell gamma ray and a stretch of lateral gamma ray, where beds may be thicker or thinner. **DTW** finds the alignment between them that minimises total mismatch, allowing local stretching and compressing.

Build a cost matrix `C[i,j] = |a_i − b_j|`. Then fill an accumulated-cost matrix:

```
    D[i,j] = C[i,j] + min( D[i−1,j], D[i,j−1], D[i−1,j−1] )
```

and backtrack from `D[n,m]` to recover the alignment. It is dynamic programming again — the same "best path through a grid" structure as everything else in this Part.

3rd place used a **Local-DTW HMM**: *"Local GR-pattern matching with small stretch."* Restricting how much stretch is allowed is essential — unconstrained DTW will happily align anything to anything.

4th place's Alijs used a **"non-ML global stretch-squeeze DP decoder"** as a *second opinion*, and fed the disagreement between it and the particle filter to LightGBM as a feature. Again: not as a predictor, as a **diagnostic**.


---

# PART V — TURNING GEOLOGY INTO A PICTURE
## The central representational trick, and everything that follows from it

If you remember one *engineering* idea from this competition, remember this one. Every single top-five solution uses it, in one form or another.

---

## Chapter 29. The problem with the obvious approach

The obvious approach is 1D sequence regression: feed the well's `(MD, X, Y, Z, GR)` sequence into a sequence model, output `TVT` at each step. Some models must also somehow consume the typewell.

It works — 3rd place's neural networks are essentially this, and they are good. But it has two structural weaknesses:

1. **The typewell is awkward to consume.** It lives on a different axis (TVT, not MD). Cross-attention can bridge it, but the model has to *learn* the matching operation from scratch, from 773 examples.
2. **Matching is a comparison, and a 1D model has to do it implicitly.** "Does the current GR look like the typewell at TVT = 7?" requires holding a whole reference curve in the hidden state and comparing against it. Hard.

There is a representation in which both of these become trivial.

---

## Chapter 30. The matching canvas

### 30.1 The construction

Build a 2D image:

- **horizontal axis = MD** (position along the well)
- **vertical axis = TVT hypothesis** (which layer you might be in)

At pixel `(row = TVT hypothesis h, column = MD position m)`, put a number that answers:

> *"How well does what we measured at MD = m agree with the reference barcode at TVT = h?"*

The simplest such number is the mismatch:

```
    canvas[h, m]  =  | GR_observed(m)  −  GR_typewell(h) |
```

Now look at what has happened. **The true TVT path is a curve drawn across this image**, and it passes through the *dark* (low-mismatch) pixels. The problem has become:

> **Find the bright ridge / dark valley that snakes across this picture from left to right.**

```
   TVT
    ↑ ┌───────────────────────────────────────────┐
      │ ░░▓▓░░░░░▓▓▓░░░░░░░▓▓░░░░░░░░░░▓▓░░░░░░░░ │
      │ ░░░░▓▓░░░░░░░░▓▓░░░░░░░░░░▓▓░░░░░░░░▓▓░░░ │
      │ ██░░░░██████░░░░░░████░░░░░░░░░░░░░░░░░░░ │  ← dark = good match
      │ ░░████░░░░░░████████░░░████████░░░░░░░░░░ │
      │ ░░░░░░░░▓▓░░░░░░░░░░░░░░░░░░░▓▓░░░▓▓░░░░░ │
      └───────────────────────────────────────────┘
        PS ──────────────► MD
        
        the answer is the path through the black pixels
```

**This is a computer vision problem now.** Specifically it is dense prediction / segmentation / keypoint tracing, and there are decades of architectures, pretrained backbones, and augmentation techniques for exactly that. That is the trick.

Notice what the representation gives you for free:

- The typewell comparison is **pre-computed into the input**, so the CNN never has to learn to compare — it just has to learn to trace.
- **Ambiguity is visible.** Multiple parallel dark bands = multiple hypotheses. The picture *shows* the multimodality, so a model that outputs a picture can express it.
- **CNNs are translation-equivariant in both axes**, which matches the physics: a pattern means the same thing at any MD, and a match means the same thing at any TVT level.
- **Convolution across the TVT axis** naturally implements "match this shape at a slight offset".

### 30.2 Who used which grid

| Team | Vertical axis | Horizontal axis | Size |
|---|---|---|---|
| **1st** | typewell TVT, ±100 ft at 0.5 ft | MD downsampled ×32 | 400 × 345 |
| **2nd** | typewell level, ±128 ft at 0.5 ft (state grid 2 ft) | MD at 32 ft/column (also a 16 ft variant) | 512 × 336 → state (21, 128, 336) |
| **4th (Lightsource)** | residual TVT axis | MD | 384 × 512 (crop 256 × 256) |
| **5th** | TVT bins | 12 ft/column | 256 × 768 (+48 appendix) |

Two design tensions run through that table:

- **MD resolution vs cost.** Finer columns = more compute. 2nd place chose coarse 32-ft columns and then recovered the lost resolution with **MD-phase TTA** at 8 sub-column offsets (Chapter 21.3) — a clever trade. They also ensembled a 16-ft/column model.
- **TVT window vs coverage.** A ±64 ft window misses wells that wander further. 5th place solved this adaptively: predict every well on a 256-row (±64 ft) canvas, flag any well whose predicted path exceeds 55 ft of relative TVT, and **re-run only those wells on a 384-row (±96 ft) canvas**, replacing their predictions entirely. Cheap where it can be, expensive only where it must be.

### 30.3 Anchoring: everything is relative

None of these canvases uses absolute TVT. They all use TVT **relative to the last known value at PS**.

Why: absolute TVT varies enormously between wells (different fields, different marker depths). Relative TVT is centred at zero for every well, by construction. Same information, dramatically smaller range, so the network is not wasting capacity learning "this well happens to be at 1,340 ft".

1st place: *"all TVT values are relative to the last visible TVT, which serves as the anchor."*
3rd place's Last-PS NN: *"target = TVT − last_known_tvt... Predicting an anchor-relative offset instead of absolute TVT removes much of the well-to-well TVT-level variation."*

This is a general principle: **predict the residual from the best thing you already know.** It shows up again in the next section.

---

## Chapter 31. The channels: what everyone actually puts in the picture

A canvas is not one image, it is a stack of images ("channels"), like a photograph's red/green/blue but with a dozen or more layers. Here is the union of what the top five used, organised by purpose.

### 31.1 Matching channels — the core signal

| Channel | Definition | Notes |
|---|---|---|
| **GR mismatch (typewell)** | `GR_typewell(h) − GR_obs(m)`, or its absolute value | The heart of it. 2nd place's `x0`, 5th place's ch9, 1st place's `|typewell_GR − horizontal_GR|`. |
| **GR mismatch (pre-PS self-reference)** | same, but against a level→GR curve built from **this well's own known prefix** | 2nd place's `x6`, added late in the competition. Higher resolution, correctly calibrated (Chapter 3.4). |
| **Coverage mask for the self-reference** | where the pre-PS reference is valid | 2nd place's `x7`. Always pair a derived channel with its validity mask. |
| **Median-based mismatch** | mismatch computed with the column *median* GR instead of the mean | 5th place's ch12/13 — median is robust to spikes. |
| **Context typewell GR** | reference typewell of the ±800 ft neighbourhood | 5th place's ch10/11 — pooling from nearby wells. |

### 31.2 Raw broadcast channels

Rather than only handing the network the mismatch, hand it the ingredients too, so it can compute its own comparisons.

| Channel | Definition |
|---|---|
| **typewell GR** | each row's reference value, repeated across all columns → **vertical stripes** |
| **lateral GR** | each column's observed value, repeated down all rows → **horizontal stripes** |
| **GR validity / coverage** | fraction of valid samples per column |
| **typewell coverage** | which rows have a valid reference |

2nd place's `x1`–`x4`, 5th place's ch0–ch2.

### 31.3 Geometry and physics channels — where equation (★) enters

| Channel | Definition | Which team |
|---|---|---|
| **row coordinate** | the TVT value of each row, relative to the anchor — a vertical gradient | 5th ch4; 1st's "TVT" |
| **dip z** | per-column `dz/dMD` — the trajectory's vertical slope | 2nd `x8`, 5th ch6 |
| **cumulative −ΔZ** | where the path would sit **if the layers were perfectly flat** | 5th ch5 |
| **azimuth sin / cos** | the well's compass heading | 5th ch7/8 |
| **known-TVT ridge** | the confirmed pre-PS path drawn as a Gaussian line into the first columns | 2nd `x4`, 5th ch3 |
| **known-column flag** | which columns are before PS | 2nd `x5` |
| **z_diff** | change in Z | 1st |
| **topography prior + its uncertainty** | 4th (James) puts the *probability that true TVT exceeds this row*, per the tabular slope model, in the green channel |

That last one is the most interesting channel in the competition. James's green channel encodes not a value but a **probability from a completely different model**, so the CNN can see both the prior and how confident that prior is, pixel by pixel. If the tabular model was sure, the green gradient is sharp; if unsure, it is smeared. The CNN learns to lean on it when it is sharp and ignore it when it is not.

### 31.4 Cross-model channels — feeding one model's output to another

| Channel | Which team |
|---|---|
| **2D particle-filter probability heatmap** | 1st |
| `\|typewell_TVT − PF_TVT\|` | 1st |
| **XY-neighbour predicted TVT, its difference, and \|typewell_TVT − predicted_TVT\|** | 1st |
| **geometric prior features** | 4th (Lightsource) |

**Caution, and 1st place's fix.** Feeding a strong model's output as a channel invites **shortcut learning** — the CNN copies the channel and stops reading the gamma ray. 1st place's countermeasure was "PF rotate/shift" augmentation: *deliberately corrupt the particle-filter channels during training* so they are unreliable and the network must keep using the raw signal. If you ever feed one model's output into another, steal this.

### 31.5 An appendix panel: a second view of the same data

5th place attached a 256 × 48 side panel to the right of the main canvas showing the typewell GR and the pre-PS lateral GR plotted on **TVT × GR axes** — a crossplot rather than a log.

They then used it as an **auxiliary task**: the model must *paint* the trajectory the lateral gamma ray would trace on that crossplot if the predicted TVT path were correct.

Their reasoning is worth quoting because it is a lovely piece of insight:

> *"When you plot the horizontal-well GR on the TVT × GR chart using the correct path versus a wrong predicted path, the failure is sometimes visually obvious. I wanted the model to learn exactly that."*

Worth about 0.1 ft. The general pattern — *find a view in which your errors are obvious, and make the model reproduce that view* — is a strong auxiliary-loss design principle.

### 31.6 A worked canvas builder

```python
import numpy as np

def build_canvas(gr_lat, z, md, tw_tvt, tw_gr, anchor_tvt, known_tvt,
                 n_rows=256, row_step=0.5, col_ft=12.0):
    """Returns (C, n_rows, n_cols). A compact version of the 5th-place layout."""
    n_cols = int(np.ceil((md[-1] - md[0]) / col_ft))
    rows   = (np.arange(n_rows) - n_rows // 2) * row_step        # relative TVT, ft
    cols   = md[0] + (np.arange(n_cols) + 0.5) * col_ft

    # per-column aggregates of the lateral log
    ci      = np.clip(((md - md[0]) // col_ft).astype(int), 0, n_cols - 1)
    gr_mean = np.array([np.nanmean(gr_lat[ci == c]) if (ci == c).any() else np.nan
                        for c in range(n_cols)])
    cover   = np.array([np.mean(~np.isnan(gr_lat[ci == c])) if (ci == c).any() else 0.0
                        for c in range(n_cols)])
    z_col   = np.array([z[ci == c].mean() if (ci == c).any() else np.nan
                        for c in range(n_cols)])

    tw_at_row = np.interp(anchor_tvt + rows, tw_tvt, tw_gr)      # reference barcode

    C = np.zeros((14, n_rows, n_cols), dtype=np.float32)
    C[0] = tw_at_row[:, None]                                    # vertical stripes
    C[1] = np.nan_to_num(gr_mean)[None, :]                       # horizontal stripes
    C[2] = cover[None, :]
    C[4] = rows[:, None] / 64.0                                  # row coordinate
    C[5] = -(z_col - z_col[0])[None, :]                          # cumulative -dZ
    C[6] = np.gradient(np.nan_to_num(z_col))[None, :] / col_ft   # dip z
    C[9] = C[0] - C[1]                                           # the mismatch image

    # channel 3: the known pre-PS path drawn as a Gaussian ridge
    for c, tv in known_tvt.items():                              # {col_index: tvt}
        C[3, :, c] = np.exp(-0.5 * ((rows - (tv - anchor_tvt)) / 1.5) ** 2)
    return C
```

---

## Chapter 32. Output heads: five ways to say "the path is here"

The canvas is the input. What comes out? This is where the five solutions differ most, and each choice has a matching decode rule and loss.

### 32.1 Dense per-pixel probability → column-wise softmax

Output one logit per pixel. Softmax **down each column**, giving a probability distribution over TVT at every MD position. Decode by expectation over rows.

Used by 1st place (2D alignment with cross-entropy, normalised along the typewell dimension) and 5th place (`head2: row-classification logits`).

**Loss:** cross-entropy against a smoothed target distribution centred on the truth.

### 32.2 Cumulative boundary mask

Output, per pixel, the probability that the true boundary is *below* this row. That makes each column a monotone "staircase" from 1 down to 0, with the boundary at the step.

Decode by simply summing:

```python
prob = torch.sigmoid(logits)          # (H, W)
row  = prob.sum(dim=0) - 0.5          # where the staircase crosses
residual = interp(row, residual_axis)
tvt = geometric_prior + residual
```

4th place's Lightsource called this mode `cum_residual`. **Loss:** `BCE + Dice/Jaccard + row_loss(decoded boundary) + optional smoothness`.

Why it is nice: summing a monotone column is exactly integrating a step function, so the decode is differentiable, cheap, and automatically sub-pixel accurate.

### 32.3 Signed Distance Field (SDF)

Instead of a probability, output at each pixel the **signed vertical distance** to the true boundary:

```
    sdf(row, col)  =  row_residual  −  true_residual(col)
```

Positive above, negative below, zero on the boundary. Every pixel now carries useful gradient information — including pixels far from the boundary, which in a binary mask carry almost none.

The decode is where it gets interesting. The obvious thing is to find the zero crossing. **Do not.** 4th place's Arunodhayan measured that a soft-expectation decode beats zero-crossing by **1.70 RMSE** — the largest single reported gain in any of the five writeups.

```python
p = torch.exp(-sdf.abs() / tau)        # tau ≈ 0.08
p = p / p.sum(dim=0, keepdim=True)     # per column
row = (torch.arange(H)[:, None] * p).sum(dim=0)
```

Their explanation: *"Soft expectation, not the zero crossing. Zero-crossing decode is worth +1.70 — it is unstable wherever the field is flat."* Where the model is unsure, the SDF is flat, and the zero crossing jitters wildly; the soft expectation degrades gracefully into "the average of the plausible region".

**Loss:** `sdf_regression + sign + row_loss(decoded)`, with a Jaccard component and a Gaussian-NLL row loss.

3rd place's "Adaptive 1D SDF" is the 1D cousin: encode the lateral and the typewell with TCNs, build a cosine-similarity correlation volume over 256 TVT candidates, fuse in trajectory and rate costs, softmax with temperature 1.75, output the expectation.

### 32.4 Control-point offsets

Instead of a value per column, predict `K` numbers — offsets at `K` control points spread along the well — and linearly interpolate between them.

4th place's James: 12 control points during pretraining, reduced to **8** for real-well fine-tuning.

Why it works here: the true TVT path is *smooth* (equation ★ again — the structural slope is nearly constant). A smooth curve is well described by a handful of knots. And drastically reducing the output dimension is a powerful regulariser when you have 700 training examples. The trade-off is that you cannot represent a sharp fault jump between control points.

### 32.5 Per-anchor conditional move distribution

2nd place's AnchorCNN. At every grid cell `(level i, column j)` output a distribution over 21 discrete moves `{0, ±2, …, ±20} ft`.

This is different in kind from the others. The other four heads output *"where is the path?"*. This one outputs *"if you were here, where would you go next?"* — for **every** possible "here", simultaneously.

Consequences:
- Training is **teacher-forced**: only the anchor lying on the ground-truth path at each column receives a loss (cross-entropy against the quantised true move, plus an auxiliary intra-bin position loss for sub-bin precision).
- Inference is the DP of Chapter 27 — the model never had to see a whole path at once, but the decoder assembles one exactly.
- It is a genuine conditional probability model `P(ΔTVT | TVT)`, so ambiguity is represented explicitly and survives all the way to the final expectation.

### 32.6 Multi-trajectory (MTP) head

Output `M` complete candidate paths plus a probability for each; train with the min-over-modes loss of Chapter 15.6; predict the probability-weighted average. 4th place's James used `M = 3`.

### 32.7 A comparison

| Head | Output shape | Decode | Represents ambiguity? | Handles faults? | Used by |
|---|---|---|---|---|---|
| Column softmax | H × W | expectation over rows | ✅ fully | ✅ | 1st, 5th |
| Cumulative mask | H × W | sum the column | partly (blurred step) | ✅ | 4th (Lightsource) |
| SDF | H × W | soft expectation | partly | ✅ | 4th (Arunodhayan), 3rd (1D) |
| Control points | K | interpolate | ❌ alone | ❌ | 4th (James) |
| Anchor moves | S × W × 21 | DP expectation | ✅ fully & explicitly | ✅ (via the move vocabulary) | 2nd |
| MTP | M × K + M | weighted average | ✅ discretely (M modes) | ✅ | 4th (James) |

James combines rows 4 and 6 — control points *inside* an MTP head — which recovers ambiguity representation while keeping the low output dimension.

---

## Chapter 33. Decoding, drift, and re-anchoring

A model predicting an 8,000-ft path in one shot will drift. Small biases integrate. Two remedies appear.

### 33.1 Chunked autoregressive decoding with re-anchoring

4th place's Lightsource:

1. Build the canvas from the *current* known TVT.
2. Predict the next ~500–950 ft of MD only.
3. **Append those predictions to the known prefix**, and rebuild the prior and the canvas from the extended prefix.
4. Repeat to the end of the well.

The original known prefix is hard-constrained and never overwritten. This keeps the model always looking at a "fresh" short-range problem instead of an ever-longer extrapolation.

Cost: errors can compound if a chunk goes wrong (there is no going back). Benefit: no long-range drift.

### 33.2 Re-anchoring as TTA

4th place's James used the same idea *without* changing the model: predict the whole well; then pretend the 25%-through prediction was known, re-render, predict the last 75%; average the two. Worth 0.18 CV.

He tested more re-anchor points: adding one at 50% helped CV but hurt private LB; a third at 75% hurt both. He kept one. **Note the discipline** — an idea that helps CV but hurts private is exactly the kind of thing to be suspicious of, and he stopped.

### 33.3 Physical constraints on the decode

3rd place clamps the HMM output:

- the formation rate is projected within the 1st–99th percentile range estimated from the last 256 known rows;
- the absolute rate is capped at 0.25;
- the **integrated** TVT correction is limited to 10 ft.

That last one is important design: it constrains total drift without forbidding legitimate local motion. Capping per-row movement would prevent the model from tracking a real steep dip; capping the integral only prevents systematic runaway.

Interestingly, 4th place's Arunodhayan measured that *clipping TVT to the typewell range* **hurt** (+0.29). Not every physical constraint helps. Constrain what actually drifts; do not constrain what the data legitimately does.

---

## Chapter 34. Multimodality, one more time — the intellectual centre

Let us gather the thread that runs through the whole book.

**The observation.** Gamma ray barcodes repeat. Given a stretch of lateral log, several TVT hypotheses fit almost equally well. The posterior over TVT is genuinely **multimodal**.

**The failure this causes.** 2nd place: *"a model that regresses a single teacher path indeed collapses onto the average of these modes — a path that corresponds to none of the hypotheses — and is therefore mismatched with the task."*

Read that carefully, because it contains an apparent paradox. Chapter 10.3 said the average of the modes is the *right* answer for RMSE. So why is collapsing onto the average bad?

**The resolution.** There is a difference between:

- (a) a model that *knows* there are two modes at ±10 and reports their mean, 0 — a well-calibrated hedge that also knows its own uncertainty; and
- (b) a model that was trained to regress a number, never learned that modes exist, and produces a mushy 0 that is the average of "everything that happened in training in vaguely similar situations".

Model (a) is right for the right reason and degrades gracefully. Model (b) is right by accident in this case and will be badly wrong in the many cases where the modes are not symmetric — when it is 80% likely to be at +10 and 20% at −10, (a) reports +6 and (b) still reports something mushy.

**Hence the universal design rule of the top five:**

> **Model the distribution explicitly. Collapse it to a point estimate only at the very last step, by taking an expectation.**

Six implementations of the same rule:

| Team | The explicit distribution | The collapse |
|---|---|---|
| 1st | 2D probability map over (MD, TVT), cross-entropy trained | expected TVT from the map |
| 2nd | `P(ΔTVT \| TVT)` at every anchor, 21 classes | exact DP marginalisation, eq (5)–(6) |
| 3rd | HMM posterior + particle cloud + Gaussian NLL heads | posterior mean, weighted mean, μ |
| 4th (James) | 3 MTP trajectories + probabilities | probability-weighted average |
| 4th (Arunodhayan) | SDF converted to `exp(−\|sdf\|/τ)` | soft expectation over rows |
| 5th | row-classification logits over TVT bins | expectation |

**And the corollary that 2nd place drew, which is the best practical justification of the whole approach:**

> *"Rather than forcing the ambiguity to resolve prematurely, this approach keeps it alive as a probability distribution over the candidate moves. Where the GR matching is ambiguous, the model can present alternative path hypotheses and flag low-confidence intervals for human review. And if independent external data (e.g., offset wells or seismic) becomes available, it can be used to reweigh the hypotheses."*

For an actual geosteering tool, that property is worth more than the RMSE.


---

# PART VI — MAKING FAKE WELLS
## Synthetic data: the thing that separated the top from the middle

773 wells. Two hundred million parameters. The arithmetic does not work.

Four of the top five solved it the same way: **generate hundreds of thousands of fake wells from a physical model of how gamma ray logs arise, pretrain on those, and fine-tune on the real 773.** 5th place went further and made synthetic data the centre of their entire strategy.

---

## Chapter 35. Why it works, and the one assumption it rests on

### 35.1 The generative assumption

Everything rests on one equation, which 2nd place calls the **layer-cake idealisation**:

```
    GR_observed(MD)  =  f( TVT(MD) )  +  r(MD)                          ...(2)
```

- `f` is the typewell's TVT → GR profile — the reference barcode;
- `r` is residual noise (everything the barcode does not explain).

*"Rocks at the same stratigraphic depth share the same properties."*

Now notice the remarkable property 2nd place points out: **(2) is a function of TVT.** So *given any TVT trajectory you like*, you can read out a consistent gamma ray log. You do not need real data to make a new well — you need a barcode and a path, and both can be sampled.

### 35.2 The recipe, in its simplest form

```
1. get a reference barcode  f  (real or synthetic)
2. invent a TVT path        TVT(MD)
3. read out the log         GR = f(TVT(MD))
4. add realistic noise      GR += r
5. derive the geometry      Z = z_layer − TVT     (equation ★, backwards!)
6. write out a well file
```

Step 5 is elegant and worth dwelling on. You do not need to simulate drilling. Pick the geology (`z_layer`) and the stratigraphic path (`TVT`), and equation (★) **hands you** a physically consistent trajectory `Z`. Every synthetic well is guaranteed to satisfy the master identity by construction, because you built it that way.

2nd place made this a design rule for augmentation too: *"while transforming trajectories, `z_layer` — not `z` — is kept as the internal representation, and `z` is derived at the very end as `z = z_layer − TVT`. This keeps identity (1) consistent by construction, no matter what transform is applied."* Choose your internal representation so that the invariants you care about cannot be broken.

### 35.3 A minimal generator

```python
import numpy as np
rng = np.random.default_rng(0)

def make_synthetic_well(master_gr, master_depth, tvt_path_bank, residual_bank,
                        n_rows=5000, md_step=1.0):
    # --- 1. barcode: crop a window from a real "master" typewell series
    lo = rng.integers(0, len(master_depth) - 900)
    tw_depth, tw_gr = master_depth[lo:lo+900], master_gr[lo:lo+900]

    # --- 2. path: mix two real TVT trajectories (2nd place's eq. 3)
    a, b = (tvt_path_bank[rng.integers(len(tvt_path_bank))] for _ in range(2))
    lam  = rng.beta(2, 2)
    n    = min(len(a), len(b), n_rows)
    tvt  = lam * a[:n] + (1 - lam) * b[:n]
    tvt  = tvt - tvt[0] + rng.uniform(tw_depth[100], tw_depth[-100])   # place it in the band

    # --- 3+4. read out the barcode, then add a real well's residual sequence
    gr = np.interp(tvt, tw_depth, tw_gr)
    r  = residual_bank[rng.integers(len(residual_bank))][:n]
    gr = gr + r

    # --- 5. derive Z from the geology via (★): pick the structural surface first
    md        = np.arange(n) * md_step
    slope     = rng.normal(0, 0.0025)                     # gentle structural dip
    z_layer   = slope * md
    if rng.random() < 0.10:                               # ~10% of wells get a fault
        k = rng.integers(n // 4, 3 * n // 4)
        z_layer[k:] += rng.normal(0, 6.0)
    z = z_layer - tvt                                     # <- (★), backwards

    # --- 6. realistic missingness
    if rng.random() < 0.3:
        s = rng.integers(0, n - 200); gr[s:s+rng.integers(20, 200)] = np.nan
    return dict(md=md, gr=gr, tvt=tvt, z=z, tw_depth=tw_depth, tw_gr=tw_gr)
```

---

## Chapter 36. Where the pieces come from

### 36.1 Barcodes: the master-series discovery (2nd place)

The single most useful data-mining finding of the competition. A forum participant noticed that the 773 typewells were not independent; 2nd place verified it and consolidated them into **54 master systems** — long continuous TVT→GR series, of which every individual typewell is a cropped window.

Once you know that, you can **crop arbitrary windows** at arbitrary depths and get a valid, real-looking typewell profile for any level range you like. Unlimited barcodes, all statistically real.

### 36.2 Paths: deform real trajectories, don't invent new ones

2nd place is emphatic: *"Feeding curves that could never appear in train makes the model learn spurious features, so instead of free-form curves, trajectories are generated in two steps from real train trajectories."*

**(a) Shape.** Deform real trajectories by vertical shift and mixup of two of them:

```
    TVT_mix  =  λ·TVT1  +  (1 − λ)·TVT2                                  ...(3)
```

**(b) Re-skinning (placement).** Assign the shape to a typewell system, and decide *at what depth* to place it. The clever bit: compute the **quantile** the original trajectory occupied within the *drilling band* of its source system (the range of levels where wells of that system are actually drilled), and map it to the same quantile of the target system's drilling band.

The shape is preserved; the depth becomes realistic for the new system; the gamma ray is regenerated from the new system's master series — a new "skin" on the same skeleton. Simple, and it keeps the data on the manifold of plausible wells.

1st place did something equivalent inside augmentation rather than generation: *"Randomly sample a TVT path while keeping TVT + Z unchanged. The sampled path is generated using block bootstrap on real TVT differences."* Block bootstrap = resample *contiguous blocks* of the real `ΔTVT` sequence rather than individual values, so local structure (autocorrelation) survives.

### 36.3 Noise: the part everyone spent the most time on

Naive white noise is not enough. The residual `r = GR_obs − f(TVT)` in real data has structure. 2nd place decomposed it:

```
    r(TVT)  =  δ_w(TVT)  +  ε                                            ...(4)
```

- **`δ_w`** — the **systematic, well-specific signature**: how much *this particular well's* gamma ray deviates from the typewell at each depth. Found by aggregating `r` over TVT for that well and taking the mean. This is the "personality" of a well: its tool calibration, its mud, its hole conditions.
- **`ε`** — the remainder. Crucially **not white**: it carries autocorrelation.

They injected noise two ways:
1. **Paste a real residual sequence** from a real well wholesale — carries over both `δ_w` and `ε` with all their real statistical properties, for free.
2. **Synthesise**: draw `δ_w` as a smooth random function of TVT, and model `ε` with an **AR (autoregressive) noise model** fitted to real data.

An AR(1) process is `ε_t = φ·ε_{t−1} + white noise`, which produces noise that is correlated over short distances — exactly what a real logging tool produces.

5th place's decomposition is the most detailed anywhere in the five writeups:

```
    hwGR(md)  =  tw( tvt(md) )  +  δ( tvt(md), md )  +  ε(md)
```

- `δ` — **lateral heterogeneity**: a Gaussian process over the TVT axis, times, along MD, a static component plus a slowly-swapping component driven by a long-correlation **Ornstein–Uhlenbeck** process. Their explanation of what this captures is excellent: *"when a single well revisits the same TVT, a similar residual reappears, but the similarity fades as you move away along MD."* The rock is *locally* the same but *regionally* varies. A model trained without this learns that revisiting a TVT means seeing an identical gamma ray, which is too easy and does not transfer.
- `ε` — **measurement noise**: white noise + a short-wavelength OU + a long-wavelength OU. Because *"measurement errors are serially correlated."*

(An **Ornstein–Uhlenbeck** process is a random walk with a pull back toward zero; its increments are correlated over a characteristic length. It is the standard tool for "noise with memory".)

### 36.4 🔬 5th place's latent-geology redesign

Their version 1 followed everyone else: generate the typewell, then make the lateral log from it. Then they noticed something:

> *"Looking at type well GR next to horizontal-well GR, you notice that the type well is smoothed along the TVT axis: GR variation from beds thinner than a certain thickness has been erased. Because of this, composing the horizontal GR as 'type well plus noise' can never close the gap to real data."*

This is Chapter 3.4's resolution asymmetry, taken seriously. The typewell log is a *low-pass filtered* view of the rocks. The lateral log, running along the beds, sees the thin beds the typewell erased. So the lateral is not "typewell + noise" — the lateral contains **information the typewell does not have**, and any generator that builds it from the typewell is structurally incapable of reproducing that.

Their fix: generate a **latent geology** — the true, unsmoothed rock column at 0.25 ft resolution — and derive *both* logs from it:

```
    L        =  L_lo + L_hi                    latent GR, low + high frequency bands
    tw(tvt)  =  H_tw[L](tvt)                   typewell = smoothed, coarsely sampled L
    hwGR(md) =  H_hw[L](tvt(md), md) + δ + ε   lateral  = a different view of the same L
```

Two different measurement operators `H_tw` and `H_hw` applied to one underlying truth. That is the physically correct picture, and it is a beautiful piece of modelling.

Their validation was equally good: they took eight images, half real and half synthetic, and asked two frontier AI models plus themselves to tell which was which. *"Codex and Gemini could not tell which is which. Neither can I."*

---

## Chapter 37. The realism paradox

Here is where it gets strange, and where two teams argued about it in the comments.

**4th place's James** built two generators. V3 was much more physically realistic than V0 by every statistical measure he could construct. And yet:

> *"By all metrics, the V3 data is more realistic than V0 (even without the filtering), but it's not necessarily 'better'. Models seem to converge more quickly when trained on the V0 data and the correlation between V0 training loss and real-well cross validation loss is quite good for up to 3 million training samples."*

His final recipe used **both**: pretrain on 500k simple V0 wells, then fine-tune on 37k filtered V3 wells, then fine-tune on real wells. Simple first, realistic second, real last.

**Tucker Arrants** (26th place) reported the phenomenon even more sharply in the comments:

> *"Each time I tried to make it more physically realistic, the synth pretrained model scored better and better on real val wells, with zero net improvement at the end of finetuning, often times regressing. Claude labeled this the 'Goldilocks problem'."*

**James disagreed** about the mechanism, from his own measurements:

> *"Generally, if a change to my pretraining procedure improved real well RMSE by X, then the final accuracy improvement after finetuning was roughly 0.5·X."*

So for James, better pretraining did transfer — at 50% efficiency — while for Tucker it did not transfer at all. The two of them landed in different regimes and neither fully explained why.

**What can we take from this?**

1. **Diversity may matter more than fidelity.** A simple generator that is *wrong* in a random way produces enormously varied data and teaches robust, general features. An over-tuned generator produces data concentrated on one manifold, and the model over-specialises to it.
2. **Fine-tuning is powerful.** Even ~700 real wells can bridge a large synthetic-to-real gap.
3. **Stage from simple to realistic.** James's V0 → V3 → real ordering, and Lightsource's "pretrain → post-pretrain (geo_plane) → real fine-tune", both do this. Broad coverage first, then narrow onto the real distribution.
4. **Measure the thing you care about, not the proxy.** Realism is a proxy. Post-fine-tune CV is the objective. James eventually stopped optimising realism after two weeks because *"almost all of my realism improvements were either neutral or harmful from a score standpoint."*

Interesting counter-note: **5th place** went the opposite way, made realism the centre of their strategy, and finished 5th (and would have been 3rd but for a bug at the deadline — see Chapter 43.7). So the paradox is not "realism is bad". It is "realism is not automatically good, and it is expensive."

---

## Chapter 38. Staging: pretrain, post-pretrain, fine-tune

Everyone's schedule, side by side:

| Team | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| **1st** | — (trains synthetic + real **jointly**) | — | — |
| **4th (James)** | 500k V0.1 synthetic, 6 epochs | 37k filtered V3 synthetic, 6 epochs | real wells, 48–112 epochs, 5-fold |
| **4th (Lightsource)** | synthetic pretrain (residual-on-prior) | "post-pretrain" on `geo_plane` synthetic | real-well fine-tune, then NNLS blend |
| **5th** | ~25 epochs synthetic | — | ~2 epochs real (8 run, best at 2) |
| **2nd** | — (each epoch mixes ~620 real wells with 2,048 fresh synthetic wells) | — | — |

Two different philosophies here, and it is worth naming them.

**Sequential (4th, 5th):** pretrain fully on synthetic, then fine-tune on real. Cleaner, cheaper, lets you reuse one expensive pretrained checkpoint across many fine-tuning variants — 4th place's four "trials" all initialise from the same Stage-2 checkpoint.

**Joint (1st, 2nd):** mix synthetic and real in every batch. 1st place tested both and reported: *"Training with simulated/augmented data and real data jointly worked better for me than a two-stage approach that first trains on simulation data."* Joint training never lets the model forget the real distribution, which sequential fine-tuning risks (catastrophic forgetting).

**A quantitative anchor.** 5th place's synthetic-only model — never fine-tuned on a single real well — scored **6.342 on the private leaderboard**, which would have placed roughly **15th out of thousands of teams**. Pure simulation, no real training data. That is a striking statement about how much physics you can bake into a generator.

### 38.1 Filtering synthetic data

4th place's James filtered his V3 output with a clever trick:

1. train a small EfficientNetV2-S on 10k synthetic wells;
2. fine-tune a copy of it on real wells;
3. have **both** models predict 50k new synthetic wells;
4. **discard** any synthetic well where the real-trained model's error is more than double the synthetic-trained model's error.

The logic: if a model that understands *real* wells does much worse on a synthetic well than a model that understands *synthetic* wells, that well is off-manifold — it contains structure that only exists in simulation. Throw it away.

This is a general and reusable recipe for **filtering synthetic data by agreement between a real-domain model and a synthetic-domain model**.


---

# PART VII — THE FIVE SOLUTIONS
## Every technique, dissected

Now we have all the vocabulary. Each chapter here walks one team's solution end to end, explaining every component in terms of Parts III–VI, and calling out what was distinctive.

**Final standings:**

| Place | Team / handle | Private | Public | CV | Core idea |
|---|---|---|---|---|---|
| 1 | Ruby (`w5833946`) | **5.639** | 5.980 | 4.627 | 2D alignment U-Net + cross-entropy, with particle-filter and spatial-neighbour features |
| 2 | Bilzard (`tatamikenn`) | **5.802** | 6.146 | 5.140 | AnchorCNN predicting `P(ΔTVT\|TVT)`, decoded by exact dynamic programming |
| 3 | Takoi + tereka | **5.836** | 6.043 | 5.288 | Five diverse candidates (HMM, PF, 2 NNs, SDF) fused by a per-row SoftMax gate |
| 4 | James Day, lightsource, alijs, Arunodhayan | **5.870** | 5.452 | 4.998 | Three independent pipelines, ridge-blended |
| 5 | daimaru (`daimaruk`) | **5.835**\* | 5.600 | 4.844 | Synthetic-data-centric CNN on a TVT×MD canvas |

\* 5th place's *best* configuration (v11) scored 5.835 private, which would have ranked 3rd — but a submission error 30 minutes before the deadline meant it was never among their two selected entries. Their actual selected submissions scored between 5.887 and 5.993. See Chapter 43.7.

Note how tight the range is: **0.23 ft separates 1st from 4th.** And note that public LB order bears almost no relation to private order.

---

## Chapter 39. First place — Ruby: "2D alignment"

**Result:** private 5.639, public 5.980, CV 4.627 (3 seeds × 5 folds). A solo gold, and the author's promotion to Grandmaster.

### 39.1 The framing

> *"I formulate the problem as a 2D alignment task and use cross-entropy loss as the main objective."*

That single sentence is the solution. Everything else is execution.

The alignment grid:

- **Horizontal (MD) axis:** 345 positions, downsampled by 32 from the raw 1-ft rows. Composed of 1,024 ft of visible region plus 10,000 ft of target region.
- **Vertical (typewell) axis:** 400 positions, covering ±100 ft around the last visible TVT at 0.5-ft resolution.

All TVT values are relative to the last visible TVT — the anchor (Chapter 30.3).

### 39.2 The loss stack

**Main loss — cross-entropy with a smoothed target.** The target is an *exponentially smoothed* probability distribution centred on the true alignment, normalised along the typewell dimension. So each MD column has a target distribution over the 400 TVT bins, peaked at the truth and decaying away from it. See Chapter 15.3.

**Auxiliary loss 1 — Huber on the expected TVT path.** Take the predicted probability map, compute the expected TVT per column, and apply a Huber loss to that. This is the "differentiable decode in the loss" pattern: part of the objective is computed on the *actual final answer*, not just the intermediate map. It directly optimises what the metric measures, while the cross-entropy shapes the distribution.

**Auxiliary loss 2 — GR penalty.** `mean(probability × grid_GR_gap)`. Read it: wherever the model puts probability mass, penalise it in proportion to how badly the gamma ray disagrees there. It is a soft physical prior pushing probability toward well-matching cells. *"Provides a small additional improvement."*

### 39.3 The model

- **Standard 2D U-Net** with a **ConvNeXt-Small** backbone: `timm/convnext_small.in12k_ft_in1k_384`.
- **Standard residual blocks in the decoder** — nothing exotic.
- **LayerNorm → BatchNorm.** *"Replacing LayerNorm in ConvNeXt with BatchNorm consistently works better, although BF16 training is required to avoid NaN losses."* (Chapters 20.3, 16.5.)
- **Average pooling + interpolation** for down/upsampling, which beat learnable alternatives. Simpler is better when data is scarce.

Notice the restraint. No custom architecture, no exotic attention. The innovation is entirely in the **problem formulation, the features, and the augmentation**.

### 39.4 The features

**Typewell channels:** GR; a "GR is NaN" mask; TVT.

The typewell GR is **calibrated** using the visible region: aggregate the lateral's `(TVT, GR)` pairs into TVT bins, interpolate back onto the typewell axis, and blend with the original typewell GR. This is Chapter 3.4 in action — the lateral gives a better-calibrated, higher-resolution look at the same rocks, so use it to fix up the reference.

**Horizontal-well channels** — per downsampled 32-ft bin, a *statistical summary* rather than a raw value:

| Feature | What it captures |
|---|---|
| GR mean | the level |
| GR NaN rate | data quality |
| GR standard deviation | how variable — thin beds vs uniform rock |
| GR slope | the trend across the bin |
| last − first GR | net change |
| quadratic fit coefficients | curvature of the GR within the bin |
| quadratic fit residual RMSE | how well a smooth curve describes it (i.e. roughness) |
| visible-region TVT mean | the known anchor, where available |

Downsampling by 32 would normally throw away all the within-bin structure. Instead of taking a mean and losing it, Ruby computes a **rich summary** of each bin: level, trend, curvature, roughness, quality. This is classical feature engineering rescuing information from a resolution reduction, and it is a technique worth copying whenever you must downsample a signal.

**Interaction channels:** `|typewell_GR − horizontal_GR|` (the mismatch image) and `|typewell_TVT − visible_TVT|`.

**Coordinate channel:** `z_diff` — equation (★).

**Particle-filter channels:** the 2D particle probability heatmap, and `|typewell_TVT − PF_TVT|`.

**XY-neighbour channels:** predicted TVT difference, predicted TVT, and `|typewell_TVT − predicted_TVT|`.

One subtle note they make: *"When particle-filter channels are present (which already contain accumulated TVT information), using accumulated TVT performs better than using only TVT differences."* The right target parameterisation depends on what is already in the input.

### 39.5 The particle filter

Standalone CV ≈ 7.4 RMSE. Built largely with AI coding assistance since the author was unfamiliar with particle filters. Improvements over the public baseline (Chapter 26.5):

- allowing low-probability large jumps (fault recovery);
- calibrated typewell GR;
- FFBSi smoothing;
- blending diverse configuration profiles;
- updating particles in bins of 64 samples rather than raw resolution.

### 39.6 🔬 The XY-neighbour model — a small piece of geology done properly

This is the most domain-specific piece of the winning solution, and it is a beautiful derivation.

**The assumption.** The local geological surface satisfies

```
    S  =  TVT + Z + C
```

and `S` is **locally linear** — i.e. over a small map area, the rock surface is a plane.

**The consequence.** If `S` is a plane, then moving from one location to another by `(ΔX, ΔY)` changes `S` by `aΔX + bΔY` for some slope coefficients `a, b`. Substituting:

```
    ΔTVT  =  a·ΔX  +  b·ΔY  −  ΔZ
```

**The estimation.** Fit `(a, b)` by **weighted least squares** over the `(x, y)` neighbours of the query point — other wells nearby whose TVT is known. Nearer neighbours get more weight.

That is it: a plane fit to the local structural surface, giving you a physics-based TVT prediction from geometry alone. Standalone CV ≈ **11.4 RMSE** — much worse than the neural network, but built from completely different information, which is exactly what an ensemble wants.

The implementation adds anisotropic distance, singular-case handling, and regularisation, but the author notes these are marginal over the simple formulation.

```python
import numpy as np

def xy_neighbor_dtvt(q_xyz, nbr_xyz, nbr_tvt, length_scale=2000.0, ridge=1e-6):
    """Fit S = TVT + Z ≈ a·X + b·Y + c locally by weighted least squares,
       then predict dTVT at the query point.  (1st place's formulation.)"""
    S = nbr_tvt + nbr_xyz[:, 2]                                # the structural surface
    d = np.linalg.norm(nbr_xyz[:, :2] - q_xyz[None, :2], axis=1)
    w = np.exp(-(d / length_scale) ** 2)                       # nearer neighbours matter more

    A  = np.column_stack([nbr_xyz[:, 0], nbr_xyz[:, 1], np.ones(len(S))])
    Aw = A * w[:, None]
    coef = np.linalg.solve(Aw.T @ A + ridge * np.eye(3), Aw.T @ S)   # [a, b, c]

    S_q = coef[0] * q_xyz[0] + coef[1] * q_xyz[1] + coef[2]
    return S_q - q_xyz[2]                                      # TVT = S − Z
```

### 39.7 🔬 The CV-versus-LB investigation, and the decision that won the competition

The XY-neighbour features **consistently improved local CV by about 0.3 RMSE** and **consistently hurt the public leaderboard**. This is the classic trap.

Most competitors would drop the feature. Ruby investigated instead.

First, a sanity check: *"XY-neighbour prediction alone scores 12.9 LB which lies in high prob region for 50 wells sample."* — i.e. the standalone LB score is statistically consistent with the standalone CV score, given the tiny public sample. So the feature is not broken.

Second, a **guarded fallback**: replace XY-based predictions with GR + `z_diff` predictions whenever the neighbourhood statistics exceeded the 95th percentile of the training distribution — that is, whenever the query point's neighbourhood looks unlike anything in training.

The five neighbourhood-quality statistics used:

1. mean neighbour distance;
2. 10th-percentile neighbour distance;
3. prefix/visible neighbour weight ratio;
4. `distance(query, neighbour centre) / average distance(neighbour, neighbour centre)` — is the query point *inside* the neighbour cloud, or outside it and being extrapolated to?
5. average `|cos corr((x_diff, y_diff), (x_nbr_diff, y_nbr_diff))|` — do the neighbours lie along the direction of interest, or perpendicular to it?

Statistic 4 is a genuinely good extrapolation detector: it distinguishes interpolation (safe) from extrapolation (dangerous) in a way plain distance does not.

The conclusion:

> *"None of these statistics explained the leaderboard degradation, and I believe they sufficiently describe neighbourhood quality. Therefore, I attribute the discrepancy to inconsistent labels and chose to trust the local CV, which consistently improves by about 0.3 RMSE."*

Private LB agreed. This decision — investigate exhaustively, find no legitimate cause, then trust the larger sample — is a large part of why they won.

Those same statistics were then reused as **ensemble routing features** (39.9).

### 39.8 The augmentations

**Z-shift** (the most important). Randomly sample a TVT path while keeping `TVT + Z` unchanged — i.e. hold the geology fixed and vary the trajectory through it, which by (★) is exactly the family of physically consistent variations. Paths are generated by **block bootstrap** on real `ΔTVT` values. GR is regenerated by matching the typewell using `TVT + TVT_noise`, where the noise is smoothed white noise. **Rare fault jumps are also simulated**, introducing offsets in `TVT + Z`.

**GR transform** (the other most important). `GR' = a·GR + b` on the typewell GR — *"forcing the model to rely more on shape than absolute values."* Directly attacks H5 (GR bias).

**The rest:**

| Augmentation | Purpose |
|---|---|
| reverse path (backward traversal, keeping part of the beginning visible) | double the data; the physics is direction-symmetric |
| MD stretching | different drilling rates, different bed thicknesses |
| 2D channel masking | robustness to any single channel |
| GR noise shift (shift the residual `GR − matched_typewell_GR` within a well) | realistic within-well GR drift |
| tail cropping | robustness to well length |
| sequential masking along the typewell axis | missing typewell coverage |
| typewell GR jitter (consistent across GR channels) | calibration noise |
| **PF rotate/shift** (corrupt PF channels) | **break shortcut learning off the PF channels** |

### 39.9 The ensemble

Six models, hand-weighted, with **two weight vectors** — one for general wells and one for the ~10% of wells with unreliable XY-neighbour information (routed by the statistics from 39.7). Those wells get models with **no** XY channels and a `z_diff` channel instead.

| Model | Features | Weight (general / no-XY) | CV | Public | Private |
|---|---|---|---|---|---|
| **Weighted ensemble** | — | — | **4.627** | **5.980** | **5.639** |
| 0719_V1 | default (GR, TVT, z_diff) | 0.07 / 0.40 | 5.09 | 5.648 | 6.130 |
| 0729_V3 | + particle filter | 0.00 / 0.20 | 5.53 | 6.202 | 6.768 |
| 0801_V1 | default | 0.07 / 0.40 | 5.16 | 5.723 | 5.884 |
| 0724_V1 | + XY-neighbour | 0.28 / 0.00 | 4.86 | 6.095 | 5.831 |
| 0801_V2 | + XY-neighbour | 0.28 / 0.00 | 4.80 | 6.166 | 5.937 |
| 0803_V2 | + XY + PF | 0.28 / 0.00 | 5.00 | 6.185 | 5.778 |

Two things to notice. First, **the ensemble (4.627 CV) is far better than the best member (4.80)** — a 0.17 gain from diversity alone. Second, look at the public column: every XY-model scores *worse* on public than the default models, and *better* on private. The public leaderboard was actively misleading, and only CV pointed the right way.

### 39.10 What Ruby says did not work

- **Transformer backbones.** *"CNN-based backbones consistently performed better, and only ConvNeXt contributed to the final solution."*
- **Two-stage synthetic pretraining.** Joint synthetic + real training beat pretrain-then-fine-tune (Chapter 38).
- **PF features in the final single model.** Originally worth a lot — CV 7.7 → 6.7 — but after adding more augmentations and stronger backbones, PF features no longer improved the best single model. Kept in the ensemble *for diversity only*. The author adds a thought worth remembering: *"Considering that my PF-only approach achieved around 7.4 RMSE in CV, a stronger PF-based method may still provide additional gains."* — that is, the ceiling was in the particle filter, not in the idea.
- **Fancy loss weighting schemes.** No consistent improvement.
- **Removing visually bad-labelled wells.** No stable gains. (Tempting, but the "bad" wells may just be the hard ones.)
- **Scaling to larger models or higher resolution.** No improvement — this is a *small-data* problem, not a *capacity* problem.

### 39.11 A note on process

> *"Most of the code was written with Codex using GPT-5.5/5.6... For the full implementation, I recommend giving the code to an AI coding agent and asking it to explain the details."*

Worth being honest about: the winner was AI-assisted throughout, including building the particle filter in a domain they did not know. What the *human* contributed was the problem framing, the alignment formulation, the diagnostic investigation in 39.7, and the judgement call to trust CV. That division of labour is a fair model for how this kind of work now goes.

**Code:** inference notebook at `kaggle.com/code/w5833946/submit-reproduce`; training code at `github.com/IAmAValidUsername/kaggle_ROGII_1st_place_solution_Ruby`.

---

## Chapter 40. Second place — Bilzard: AnchorCNN and conditional probabilistic path modelling

**Result:** private 5.802, public 6.146, OOF 5.140.

Of the five, this is the most conceptually clean, and it is the one to study if you want to understand the *ideas* rather than collect the tricks.

### 40.1 The data insight that shapes everything

Covered in Chapter 5.1, but restated in their words. The expert-interpreted formation surface columns (`ANCC`, `BUDA`, ...) have a characteristic **"sequence of straight lines"** shape. With `z_layer := TVT + z − b_well`:

```
    dTVT  =  dz_layer  −  dz                                             ...(1)
```

`dz_layer/dMD` is **staircase-shaped**: constant over most intervals, with occasional steps, and steps appear in only about 10% of columns.

> *"Since the identity fixes the coefficient at −1, wherever there is no step, the shape of `dTVT` matches `−dz` exactly (this is why the two curves run parallel in the figure). → By feeding `dz/dMD` (dip z) as an input feature, the model's job shrinks to 'estimating the smooth structural slope `dz_layer`', strongly steering it toward formation surfaces that experts would plausibly draw."*

**Shrinking the model's job** is the right way to think about feature engineering. Not "add information" but "remove work".

### 40.2 The synthetic data pipeline

Rests on the layer-cake assumption `GR_obs = f(TVT) + r(TVT)` (Chapter 35.1). The vertical material (barcodes) and horizontal material (paths) are sampled from **independent pools** and composed:

1. Sample a **master** from the master pool (54 systems — Chapter 36.1).
2. **Crop a window** = the typewell profile `f`.
3. Sample a **TVT trajectory shape** from the real-well pool, and place it at a depth inside the level band.
4. **Read out** the gamma ray along the trajectory: `f(TVT(MD))` — noise-free.
5. **Add a residual** `r` sampled from the residual bank (773 real wells).

Trajectory generation in two steps: **shape** by vertical shifts plus mixup `TVT_mix = λ·TVT1 + (1−λ)·TVT2`, then **re-skinning** — mapping the quantile the trajectory occupied within its source system's drilling band to the same quantile of the target system's band (Chapter 36.2).

Residuals decomposed as `r(TVT) = δ_w(TVT) + ε`, injected either by pasting a real well's residual sequence wholesale or by synthesising `δ_w` as a smooth random function plus an AR model for `ε` (Chapter 36.3).

Throughout, `z_layer` is the internal representation and `z` is derived last, so (1) holds by construction under any transform.

### 40.3 The augmentations

- **Geometric:** MD flip, level flip (with label sign flip), random MD crop, level shifts (±32 ft).
- **Structural:** flips and scaling in `z_layer` space — literally stretching the structural slope. Consistent with (1) automatically, because of the `z_layer` representation.
- **Length normalisation and stitching:** short wells are extended to a target length by copy-and-paste of themselves or fragments of other wells, with **seams smoothly joined by short splines (100 ft)**. Also a "copy" augmentation that extends a segment inside the evaluation zone to `X + reverse(X) + X` — values stay continuous at the seams, only the gradient flips.
- **Observation quality:** **SpecAugment** — masking spans of GR (up to 21 columns) *in exactly the same format as real missing data*.

That last parenthetical matters. Simulate missingness the way it actually occurs (contiguous spans, at realistic lengths), not as scattered random dropout.

### 40.4 The input tensor

A `(9, 512, 336)` image: vertical = typewell level (0.5 ft/bin, ±128 ft), horizontal = MD (32 ft/column, first 16 columns are the known pre-PS interval).

| ch | content |
|---|---|
| `x0` | **GR mismatch (typewell)** — for each (level, MD), the discrepancy between lateral GR and typewell GR under the hypothesis that the well is at that level |
| `x1` | lateral GR (broadcast over level) |
| `x2` | GR validity (broadcast) |
| `x3` | typewell coverage (broadcast over MD) |
| `x4` | **Gaussian ridge of known TVT** — the pre-PS path drawn into the first 16 columns |
| `x5` | known-column flag |
| `x6` | **GR mismatch (pre-PS self-reference)** — matching against a level→GR reference built from this well's own pre-PS interval, per the host's tip |
| `x7` | coverage mask for `x6` |
| `x8` | **dip z** — per-column slope of the trajectory |

`x6`–`x8` were added late in the competition and are the highest-value three: the self-reference (Chapter 3.4) and the physics channel (equation ★).

### 40.5 The AnchorCNN

**The design analogy.** From object detection (YOLO, CenterNet): *"CNNs are good at encoding local features, and the move direction `dTVT` at a specific coordinate (MD, TVT) is easy to predict from local features. So I mapped the conditioning part — 'at coordinate (MD, TVT)' — onto CNN anchors."*

An **anchor** is the embedding vector at coordinate `(i, j)` of the feature map. `(i, j)` = "the well is at level `i`, in MD column `j`". From each anchor, read out a move distribution conditioned on that position.

**The architecture.** EfficientNet-B0; all stage outputs fused by an FPN-style decoder (1×1 conv to a common width, interpolate to one resolution, sum); mapped to a `(B, 21, 128, 336)` state grid — vertical = level at 2 ft/bin, horizontal = 32 ft/column. The 21 channels are the **move vocabulary**: `{0, ±2, ±4, …, ±20}` ft/column, i.e. `{−10, …, +10}` bins of 2 ft.

**Why this is the natural formalisation of multimodality.** Their reading of the Alyaev et al. paper: the essence is modelling multi-path as a conditional probability model `P(dTVT | TVT)` — *"holding, for every level hypothesis simultaneously, the distribution of 'if the well is at depth TVT now, how much does it move up or down over the next interval'."*

Note what this does that a single-path model cannot: the model never has to decide *where the well is*. It answers a purely local question at every hypothesis. The global decision is deferred entirely to the decoder.

### 40.6 Training

**Teacher forcing.** At each column, only the anchor lying on the ground-truth level receives a loss: cross-entropy against the quantised ground-truth move, plus an auxiliary **intra-bin position loss** (so the model can express sub-2-ft precision within a move class).

Each epoch mixes ~620 real training wells per fold with **2,048 synthetic wells generated on the fly** (joint training, Chapter 38). 120 epochs, constant LR 1e-3 after warmup (WSD, no decay), EMA. **5 folds × 3 seeds in about 10 hours on a single GPU** — remarkably cheap for a 2nd-place solution.

### 40.7 Decoding by dynamic programming

Two options were available (Chapter 27):

- **Rollout:** sample stochastically from `P(dTVT | anchor)` to generate a bundle of paths.
- **Path expectation (DP):** marginalise the state grid exactly.

```
    p_{k+1}(s)  =  Σ_{s'} p_k(s') · P(dTVT = s − s' | anchor(s', k))      ...(5)
    TVT_hat(k)  =  Σ_s p_k(s) · TVT(s)                                    ...(6)
```

*"No sampling is involved — (5)–(6) are computed exactly."*

And the finding: *"Viterbi (maximum-likelihood path) and other mode-tracking decoders were consistently worse than the expectation."*

### 40.8 Ensembling and MD-phase TTA

Final submission averages: **5 folds × 3 seeds = 15 checkpoints** of DP paths × **MD-phase TTA (8 views)** × **2 grid resolutions** (32 ft/column and 16 ft/column). Scores 200 hidden wells in ~40 minutes on a Kaggle T4.

**MD-phase TTA**, explained in their own reasoning: the CNN's MD resolution is 32 ft per column, so there is a phase ambiguity below 32 ft — *where you cut the columns* changes the input. Relative to the gamma ray correlation length (**~18 ft**), a 32-ft box average is coarse, and the values themselves change with the cut point (aliasing). So run inference at 8 sub-resolution phases, shifting the column-grid origin by `{0, 8, 16, 24, 32, 40, 48, 56}` ft, and average.

Worth 5.624 → 5.417 OOF. This is signal-processing thinking applied to a CNN input pipeline, and it generalises to any model that bins a continuous signal.

### 40.9 🔬 Robust validation — the best model-selection protocol in the competition

Their framing:

> *"The competition metric, pooled RMSE, is proportional to the sum of squared errors, so it is easily dominated by 'catastrophic wells'. The metric moves far more when you fix a few badly failing wells than when you further improve wells that are already doing fine. The question is whether an improvement on a few wells is systematic or just luck."*

The **leave-largest-contribution-out** acceptance test is given in full in Chapter 19.6, with code. Summary: compute per-well SSE differences `g_w`, remove wells in descending `|g_w|`, and find `k*`, the number of removals at which the improvement vanishes. Reject unless `k*` is large.

Their real example: one candidate looked 0.28 better at `k = 0`, but removing just **8 of 773 wells** erased it → rejected. Another survived removing **52** wells (the size of the public LB) → adopted.

### 40.10 Results, including the two final submissions

| sub | configuration | OOF | public | private |
|---|---|---|---|---|
| 068 | AnchorCNN (32 ft/col) × 15 ckpt | 5.624 | 5.780 | 6.126 |
| 071 | 068 + MD-phase TTA8 | 5.417 | 5.820 | 6.078 |
| 075 | 16 ft/col × TTA8 alone (diagnostic) | 5.348 | 5.970 | 5.882 |
| **076 (final, selected)** | `dz_layer`-head models × TTA8 + 16 ft/col × TTA8 | **5.140** | 6.146 | **5.802** |
| 077 (final, hedge) | 068-family × TTA8 + 16 ft/col × TTA8 | 5.242 | 5.910 | 5.905 |

Again: OOF ranks the submissions correctly (5.140 → 5.802 is the best of both columns) and public ranks them backwards.

Note the hedging strategy: two final submissions, one chosen by CV (076) and one more conservative (077). The CV-chosen one won by 0.10.

### 40.11 What didn't work

**Selecting from a path bundle (reranker, RL).** A rollout bundle of `N = 1024` genuinely contains excellent paths — an oracle picking the best in hindsight beats the deployed score by several feet. They built a reranker and tried reinforcement learning (GRPO) to capture that gap. *"Both selectors overfit the evaluation data easily and never consistently beat the DP expectation."* Design fork resolved: commit to DP.

**Explicit formation modelling.** Modelling the formation surfaces from neighbouring wells captures the coarse topography but could not fix the catastrophic wells. Their diagnosis (Chapter 6): *"the teacher labels (expert interpretations) do not reference the formations directly — they are relative judgements based on GR matching."*

This is a striking disagreement with 4th place's James, whose entire pipeline is built on exactly this topography modelling and who finished 4th with it. Two competent teams, opposite conclusions. The likely resolution is that topography is a good *prior* to correct with vision (James) but not a good *fix* for catastrophic mis-correlations (Bilzard) — they were testing different uses of the same information.

### 40.12 The closing thought

> *"What this model imitates is the interpretation work of geosteering experts — and the ambiguity in that work is intrinsic to the task. Rather than forcing the ambiguity to resolve prematurely, this approach keeps it alive as a probability distribution over the candidate moves. I believe this is its most practically valuable property: where the GR matching is ambiguous, the model can present alternative path hypotheses and flag low-confidence intervals for human review."*


---

## Chapter 41. Third place — Takoi & tereka: five candidates and a gate

**Result:** private 5.836, public 6.043, fold-safe local CV 5.2884.

This is the most *engineered* solution of the five, and the one to study for pipeline discipline. It is also the only one that treats classical probabilistic models and neural networks as equal partners.

### 41.1 The strategy

> *"The core idea of our solution is to generate several physically plausible TVT trajectories, complement them with neural models that make different kinds of errors, and combine the candidates at every position using SoftMax Gating."*

Their reasoning for why neither pure approach works:

> *"Pure time-series extrapolation accumulates even a small formation-dip error over a long horizontal section. Pure GR matching is also ambiguous because similar GR patterns, GR bias, missing values, and thickness changes can produce several plausible alignments."*

Three stages: (1) match GR with an HMM and a particle filter while preserving trajectory continuity; (2) use the physics predictions plus trajectory, GR and typewell information in neural networks to correct systematic physics errors; (3) dynamically combine five complementary candidates with a SoftMax gate.

```
Horizontal well + Typewell
          |
          +--> 3-family HMM -------------------+
          +--> Particle Filter ----------------+
          +--> 49-feature Last-PS NN ----------+--> CNN / BiLSTM SoftMax Gate
          +--> 31-feature Delta NN ------------+          |
          +--> Adaptive 1D SDF ----------------+          v
                                                    5-split average
                                                          |
                                                    submission.csv
```

### 41.2 The component scoreboard

| Method | Training loss | Local CV | Public | Private |
|---|---|---|---|---|
| 3-family HMM | none (probabilistic state estimation) | 5.9703 | 6.207 | 6.229 |
| Particle Filter | none (sequential Monte Carlo) | 6.5827 | 7.011 | 6.725 |
| NN from Last PS | Gaussian NLL | 5.4723 | 6.074 | 5.903 |
| Delta NN | delta RMSE | 5.7703 | 6.093 | 6.102 |
| Adaptive 1D SDF | CE + 0.25·SmoothL1 + 0.10·delta + 0.05·RNC | 9.5415 | 8.502 | 8.837 |
| CNN SoftMax Gate | prediction MSE + best-candidate CE | 5.4255 | 6.027 | 5.814 |
| BiLSTM SoftMax Gate | same | 5.3329 | 6.043 | 5.836 |
| **Final 5-split ensemble** | same as BiLSTM gate | **5.2884** | **6.043** | **5.836** |

Look at the Adaptive 1D SDF row: **9.54 CV, by far the worst component**, and they kept it. Their justification is the essence of ensembling:

> *"Its standalone RMSE was weaker than the other candidates, but it solved the task through direct horizontal/typewell correspondence rather than physics-state estimation or direct regression. This increased candidate diversity, and the SoftMax Gate could select it locally even when its average weight was small."*

A weak model that is wrong *differently* is worth more than a strong model that is wrong the same way as everything else.

### 41.3 The HMM stack

Covered in Chapter 25.6. Recapping the essentials:

**Hidden state:** TVT position; formation rate `d(TVT + Z)/dMD`; GR bias; reference family (typewell-oriented or sibling-lateral-oriented).

**Reference GR — their #1 private-LB insight.** Training wells sharing a typewell were grouped as **siblings**, and their lateral GR curves aggregated into 0.25-ft TVT bins. The base HMM used two reference mixtures with typewell/sibling weights 0.2/0.8; the reference identity remained *part of the hidden state*. They also built a `(TVT_input, GR)` template from the target well's own known prefix and mixed it in according to observation count — *"this self-reference exploits the fact that GR calibration is shared between the known prefix and future section."*

**Initialisation:** robust formation rate from the last **256** known-prefix rows; prepend the last 128 prefix rows to the HMM sequence; clamp the prefix TVT states to the observed values.

**Emission:** Student-t with **one** degree of freedom (Cauchy) — robust to outliers.

**Transitions:** encourage smooth changes in position, rate and bias. Forward-backward smoothing produces the posterior mean.

**Output clamping:** project using the 1st–99th percentile formation-rate range from the last 256 known rows; absolute rate capped at 0.25; integrated TVT correction capped at 10 ft.

**Three families blended** 0.50 / 0.20 / 0.30 (base / Local-DTW / fine-bin), improving OOF from 6.0492 to 5.9703.

### 41.4 The particle filter as a feature extractor

Settings: 500 particles, 240 seeds, 3 seed bases, rate momentum 0.9995, rate noise 0.001, position noise 0.005; systematic resampling on low ESS; reference and bias families inherited after resampling; two observation models (`f015` — 15% of particles in a "hard" family whose likelihood is progressively strengthened near the start; `soft` — all particles allow GR bias) blended 0.6/0.4.

Seed-level paths weighted by full-sequence likelihood and smoothed by genealogy tracing.

**The features handed to the neural networks:** weighted mean; spread; skewness; IQR; q10, q25, q50, q75, q90; and the previous-row and next-row differences of all of those.

That is a complete uncertainty profile per row. The neural network can now learn "when the particle spread is large *and* skewed, the physics estimate is unreliable — override it".

The standalone PF candidate was `0.6 × f015 + 0.4 × soft`. The "HMMPF" physics feature was `0.6 × three-family HMM + 0.4 × PF`. *"The HMM contributes a smooth, stable global path, while the PF contributes alternative hypotheses and uncertainty."*

### 41.5 The two neural networks

**Neural Network from Last PS.** Anchors every hidden row at `last_known_tvt` and predicts `target = TVT − last_known_tvt`. *"Predicting an anchor-relative offset instead of absolute TVT removes much of the well-to-well TVT-level variation."*

Its **49 features**: raw/mask (`X, Y, Z, GR_filled, GR_missing, known_tvt_mask`); PS-relative geometry (`md_from_ps, z_from_ps, Z_diff_1`); the full PF distribution (f015/soft mean offsets, spread, skewness, IQR, q10–q90); local dynamics (previous/next differences for X/Y/Z/GR/TVT_input and the PF statistics); physics candidates (HMMPF offset, MultiGrid HMM offset, and their gap); and the correlation candidate (anchor-relative Adaptive 1D SDF prediction).

Note "and their gap" — the *disagreement between two physics models* is itself a feature. Disagreement is a proxy for difficulty.

**Architecture:**
- input projection to 96 channels;
- **16 full-resolution dilated-TCN residual blocks**, kernel size 7, dilation cycle 1, 2, 4, …, 128, repeated twice (Chapter 20.2);
- **MaxViT1D** local-window/sparse-grid attention after blocks 4, 8, 12, 16;
- **downsampled typewell cross-attention** after block 8 (Chapter 20.7);
- typewell encoder: residual encoder + gated BiLSTM;
- output head: gated one-layer LSTM;
- **Gaussian head** predicting mean and variance.

The design keeps **full resolution throughout** (dilation rather than pooling), which matters when you need half-foot precision over an 8,000-row sequence.

**Loss:** Gaussian NLL, `σ² = softplus(raw) + 0.001`. At inference only `μ` is used as the candidate; **σ is supplied to the gate as a confidence feature**. Training: 100 epochs, batch 16, Scheduler-Free AdamW, LR 0.002, weight decay 1e-4, dropout 0.1, EMA 0.995, with short contiguous GR blocks masked during training.

Standalone OOF 5.4723. Switching from RMSE loss to Gaussian NLL improved the three-split ensemble from 5.4176 to 5.3645.

**Delta Neural Network.** Predicts `target_t = TVT_t − TVT_{t−1}` and integrates from the known TVT at PS. *"This representation captures local slope changes well, but small biases accumulate over long distances, giving it a different error pattern from the Last-PS NN."*

Its **31 features** emphasise *differences and uncertainty*, not absolute levels — previous/next differences of X/Y/Z/GR/TVT_input; PF spread, skewness, IQR and their adjacent differences; previous/next differences of the HMMPF, MultiGrid and SDF offsets; `TVT_input_observed_delta_last`.

Same architecture family, independently trained, delta-RMSE loss. Dropout 0.5, weight decay 0.01, **manifold mixup** on the head representation (p = 0.5, α = 0.2). Standalone OOF 5.7703 — weaker, kept for complementarity. A Gaussian-NLL variant of the Delta NN made things worse, so it kept RMSE.

**This pair is a designed complement:** absolute-offset prediction (no drift, but poor local detail) versus delta prediction (excellent local detail, accumulating drift). Their errors are structurally different, which is precisely what the gate needs.

### 41.6 Adaptive 1D SDF

An independent correlation model — it does **not** consume the HMM or PF outputs, so it stays uncorrelated with them until the gate.

1. Encode horizontal GR and trajectory with a four-block TCN.
2. Encode typewell GR/TVT context with a three-block TCN.
3. Build a **query–key cosine-correlation volume** at embedding dimension 128.
4. Compare **256 TVT candidates** on the typewell.
5. Fuse GR-value cost with trajectory progress, required formation rate, and graph-edge features.
6. Estimate well-specific GR bias correlation and scale from the known prefix.
7. Softmax with temperature 1.75; output the **expected** TVT.

**Loss:** `L_candidate_CE + 0.25·SmoothL1(TVT/10) + 0.10·L_delta + 0.05·L_RNC` — cross-entropy on the nearest candidate row, regression on the expected TVT, an auxiliary delta loss, and Rank-N-Contrast to teach the embedding the *ordering* induced by TVT distance.

Steps 3–4 are a **cost volume**, the same construction used in stereo matching and optical flow: compare every query position against every candidate position and let the network reason over the resulting similarity map. It is the 1D cousin of the 2D canvas from Part V.

### 41.7 SoftMax gating — the distinctive contribution

Five candidates arrive at every row: HMM, PF, Last-PS NN, Adaptive 1D SDF, Delta NN. A small network outputs logits over them and the prediction is

```
    prediction_t  =  Σ_k  softmax(logits_t)_k  ×  candidate_{t,k}
```

**The gate's own features** are as interesting as the candidates:

- each candidate's offset from the last known TVT;
- each candidate's deviation from the candidate mean;
- one-step candidate changes;
- **candidate standard deviation and range** (i.e. how much the candidates disagree);
- MD/Z distance from PS, MD/Z steps, 3D step length;
- GR, GR difference, GR-missing flag;
- normalised position within the hidden section;
- the **Gaussian predictive σ** from the Last-PS NN.

So the gate can learn rules like "near PS, trust the physics; far from PS with high candidate disagreement and high NN σ, hedge toward the mean".

**Gate architectures:** a CNN gate (hidden 32, kernel 7, two residual blocks) and a BiLSTM gate (hidden 32, one bidirectional layer). Both trained 8 epochs with

```
    L_gate  =  MSE( y , Σ w_k · candidate_k )  +  CE( w , best_candidate )
```

The first term optimises the mixed output; the second is classification against whichever candidate had the smallest absolute error at that row. The second term is a strong learning signal that pure MSE would give only weakly.

The BiLSTM gate beat the CNN gate (5.3329 vs 5.4255 OOF), and the final deployment weights were CNN 0, BiLSTM 1.

**Why softmax and not free weights?**

> *"Fixed weights treat the area immediately after PS, long extrapolation ranges, missing-GR regions, and regions with large candidate disagreement identically. A SoftMax Gate can vary weights continuously using within-well context and candidate disagreement. Nonnegative weights summing to one also constrain the output to a convex combination. With only 773 wells, this reduced extreme extrapolation compared with a high-capacity unconstrained stacker."*

They tested the unconstrained alternative — a BiLSTM refiner that directly predicts `last_known_tvt − target_tvt` instead of mixing candidates. It scored **5.4023** versus the SoftMax gate's **5.2797**, and **lost in all five folds**. Constrained beat unconstrained decisively.

**And they name the limitation honestly:** *"a convex combination cannot correct drift shared by all candidates beyond their range. OOF oracle analysis confirmed this ceiling, so SoftMax represented a deliberate preference for robustness over maximum flexibility."*

### 41.8 The validation architecture

The six fold-safe rules are in Chapter 19.3, and the 5 splits × 5 folds design in Chapter 19.4. In numbers: five model families × 5 splits × 5 folds = **125 checkpoints**.

The measured payoffs, all on private LB:

| Change | Private LB |
|---|---|
| 3 split patterns → 5 split patterns | 5.903 → 5.836 (−0.067) |
| initial-rate window 30 rows → 256 rows | 5.854 → 5.812 (−0.042) |
| spatial-distance-weighted sibling HMM (other branch) | 5.817 → 5.691 (−0.126) |
| rational distance weighting (other branch) | 5.817 → 5.720 (−0.097) |
| Rao-Blackwellised PF bias (other branch) | 5.691 → 5.688 (−0.003) |
| fold-specific HMM reference pools (other branch) | 5.726 → 5.718 (−0.008) |

They also recomputed physics features **inside the Kaggle notebook** for the mounted hidden test data rather than depending on a precomputed table — a reproducibility choice that protects against the dataset mount or test contents changing.

### 41.9 Their own summary of what mattered

> *"Our main takeaway is to prioritise the right reference GR, stable initial-rate estimation, diverse HMM error modes, and well-level CV diversity before adding model complexity. The neural networks and SoftMax Gate were powerful, but the physics pipeline provided the foundation for Private generalisation."*

The biggest single gain in their table (−0.126) came from **how they built the reference gamma ray curve** — not from any architecture change. Domain-side data work beat model-side work.


---

## Chapter 42. Fourth place — James, Lightsource, Alijs & Arunodhayan: three pipelines, one blend

**Result:** private 5.870, public 5.452, CV 4.998. A four-person team whose members built *independent* solutions and blended them.

| Pipeline | Ensemble weight | CV | Public | Private |
|---|---|---|---|---|
| James | 0.644 | 5.143 | 5.463 | **5.730** |
| Lightsource | 0.327 | 5.570 | 5.897 | **7.045** |
| Alijs | 0.029 | 7.035 | 6.217 | **9.138** |
| **Full ensemble** | — | **4.998** | **5.452** | **5.870** |

Read that table carefully, because it is the most instructive table in the whole competition. **Two of the three components were near-useless on the private leaderboard** (7.045 and 9.138), yet the blend held at 5.870 — better than two of its three members and only 0.14 worse than the best. Ensemble weights fitted on OOF gave James 64% of the weight, which turned out to be the right call, and the small weights on the weak members did not sink the ship.

Their own summary: *"Only one of our three solutions performed well on the private leaderboard, but the other parts were useful in local cross validation and the full ensemble was robust enough for a cash gold finish."*

Ensemble weights were fitted by **ridge regression on OOF predictions**.

---

### 42.1 James's pipeline — topography first, vision second

Inspired by the winner of the **PhysioNet ECG digitisation** competition — a completely different domain (turning scanned paper ECG traces back into signals), but structurally the same problem: *trace a curve through a rendered image*.

**Four steps:**

**Step 1 — Predict the structural slope with tabular models.**
Predict the slope of the rock formations along the drill path, in 750-ft segments, by feeding `X, Y, Z, azimuth, pitch` to an ensemble of tabular regression models. *"It works because the training and test wells originated from the same basin with a mostly-linear global slope and deviations that can sometimes be predicted based on neighbouring wells."*

Crucially, the models produce **uncertainty estimates** so downstream vision models can tell how confident the slope estimate was.

**Step 2 — Convert slope into a TVT baseline via equation (★).**
*"We're given a last-known TVT, so if change in TVT is known, then it becomes trivial to precisely calculate the unknown TVT values. Change in TVT = change in rock elevation − change in the drill's Z coordinate. All of the Z values are given, so the only real unknown is change in rock elevation."*

Integrate the predicted slope over distance → rock elevation change → subtract the known `ΔZ` → a baseline TVT path.

**Step 3 — Render everything as an RGB image.**
Vertical columns = measured depth; horizontal rows = candidate TVTs.

| Channel | Content |
|---|---|
| **Red / Blue** | GR *residual*: is the observed GR at this MD higher (red) or lower (blue) than expected under this TVT hypothesis? |
| **Green** | the probability, per the topography model, that the true TVT exceeds this candidate value |

Splitting the residual into red (positive) and blue (negative) rather than using one signed channel gives the CNN a clean, ReLU-friendly representation. And the green channel is the *entire tabular model, with its uncertainty*, drawn as a picture (Chapter 31.3).

**Step 4 — Predict control-point offsets with a vision ensemble.**
ConvNeXt V2-Large with feature-pyramid regression heads predicts **8 control-point offsets** to shift the topography-based path up or down; offsets between control points are linearly interpolated. Training in three stages: V0 synthetic → V3 synthetic → real wells.

#### The topography sub-model, in detail

~90% of the vision-model weight received predictions from relatively simple **LightGBM quantile regression** models predicting upper and lower confidence intervals for `dA/dM` (the derivative of the ANCC surface with respect to MD) from the segment midpoint coordinates plus azimuth and pitch.

The other ~10% received predictions from a stronger ensemble of **TabPFN + RealMLP + LightGBM** predicting `dA/dM`, `dA/dX`, `dA/dY`. That stronger ensemble is substantially better in isolation — slope RMSE 0.0100 → 0.0088; private LB *without* vision models 12.6 → 10.8; correlation between predicted σ and real error 0.31 → 0.40.

**And he mostly did not use it.** Why?

> *"The CV gains of the topography-only predictions were ~20× larger than the public LB gains, and finetuning models on these predictions caused them to focus 'too heavily' on the topography outputs, thereby making the public LB scores worse. Furthermore, the vision models are somewhat sensitive to changes in how the confidence intervals are calibrated."*

This is the shortcut-learning problem (Chapter 31.4) again. Give the vision model a *very good* prior and it stops reading the gamma ray. A deliberately *weaker* prior keeps the vision model honest. 1st place solved the same problem by corrupting the channel; James solved it by keeping the channel weak.

He did include one model that received the strong topography: a **foundation model trained purely on synthetic data with no real-well fine-tuning**, precisely because it *"prevent[s] it from 'overfitting' to any particular topography model's behavior."*

#### The vision architecture

All models shared one architecture; diversity came from augmentation, LR schedule, topography choice, data bagging, and the frozen foundation model — **not** from architectural variation.

- `convnextv2_large.fcmae_ft_in22k_in1k_384`
- feature-pyramid head using **all four** backbone stages, 16 projected channels per level, 32 hidden channels
- **three candidate trajectories with MTP mode probabilities** (Chapter 15.6)
- 384 × 384 inputs, ±75 ft TVT window
- no topography centering, no ImageNet input normalisation
- BF16, AdamW, no head dropout
- 12 control points in pretraining, 8 in real-well fine-tuning

**On backbone scale:** *"Small (~50M parameter) models seemed to be optimal with only 5K synthetic pretraining wells and only ~700 real wells for finetuning, but with small models there was little-no score improvement when scaling from tens of thousands of pretraining wells to hundreds of thousands. ConvNeXt Large with ~540k training wells works well enough that the small models are no longer useful."* Model capacity and synthetic-data volume have to scale together — a big model on a small synthetic corpus is wasted, and a small model cannot exploit a big one.

**On heads:** shallow, few channels, forcing the backbone to do the work. For mode-probability selection, **mean and max pooling concatenated** beat either alone.

**On the MTP loss:** trained on a hybrid of MTP loss and weighted-average-trajectory loss, because *"optimizing purely for the MTP loss function proposed in that paper is somewhat suboptimal in terms of minimizing single-path RMSE."* And the scaling failure mode (Chapter 15.6): past ~3M training samples ConvNeXt V2-Large began ignoring the classification term entirely to minimise the weighted-average term.

#### The synthetic generators

**V0 / V0.1** (500k wells) — *"TVT → expected GR → residual/GR → BUDA → derive Z"*:
sample last-known TVT and prediction length from a bank of real wells; generate a synthetic typewell GR-vs-TVT window; generate a TVT-delta path from the empirical path bank; interpolate the typewell at those TVTs for expected GR; generate residuals conditioned on expected GR (an expected-GR-dependent component plus low-, medium- and high-frequency noise); sample a single BUDA slope and build a linear BUDA path; back-calculate `Z = BUDA − TVT`. Topography errors are piecewise-constant Gaussian slope errors calibrated to a requested integrated TVT RMSE.

**V3** (37k filtered wells) — *"BUDA → conditioned Z → derive TVT → expected GR → residual/GR"*:
build a BUDA path with a sampled starting level and global slope, **correlated slope wander**, and **explicit fault jumps**; generate `Z` conditioned on that path (modelling `dZ/dMD` from the smooth BUDA increments plus correlated residual variation); derive `TVT = BUDA − Z + const`; resample if TVT falls outside the typewell range; interpolate for expected GR; generate a residual conditioned on the expected-GR spine. V3's residual generator uses expected GR much more thoroughly — expected level and local contrast influence both the mean residual and its local variance, followed by correlated fat-tailed noise, coherent excursions, and missingness. Its topography errors are sampled empirically per segment rather than as global Gaussians.

Note the reversal of causal direction between V0 and V3. V0 picks the stratigraphic path and derives the geometry; V3 picks the geology and geometry and derives the stratigraphic path. V3 is the physically correct order.

V3 was **filtered** by the two-model agreement trick of Chapter 38.1.

His candid note on process: *"My main role in generator development was torturing GPT-5.5 and Opus-4.8 with a ton of complaints about statistical discrepancies between the real and synthetic data."*

#### The fine-tuning ensemble

| Trial | Role | Epochs/batch | Schedule | Loss | CV | Weight |
|---|---|---|---|---|---|---|
| 0 | frozen Stage-2 foundation model, V2 topography | — | — | — | 5.9270 | 11.15% |
| 1 | baseline finetune | 64 / 8 | cosine, 1.027e-4, 15% warmup | Huber 9.48 | 5.2940 | 16.81% |
| 8 | affine-heavy variant | 80 / 16 | linear, 3.699e-5, 30% warmup | MSE | 5.2842 | 3.20% |
| 17 | **best individual** | 112 / 8 | cosine, 3.284e-5, 15% warmup | Huber 0.549 | **5.2013** | 48.50% |
| 25 | shorter, low-LR | 48 / 16 | linear, 1.987e-5, 10% warmup | Huber 3.013 | 5.2826 | 20.33% |

Their augmentation recipes differ deliberately (Trial 1 reuses the pretraining recipe; Trial 8 swaps flips for strong GR-affine transforms; Trial 17 augments throughout all 112 epochs; Trial 25 uses strong lateral affine and small vertical masks), as do the MTP loss weights (classification weight 0.164 / 0.0625 / 0.485 / 0.0898).

**Blend result:** after optimal TTA and quarter-point re-anchoring, 5.2013 → **5.1428**, a 0.0585 gain.

The **Stage-1 augmentation recipe** in full, for reference: mixup 75% (α = 0.624); vertical/horizontal image flips 50%/25%; GR-residual inversion and residual flip 25% each; simulated vertical shift 25%; typewell flip during shifting 25%; topography-error permutation 25%; lateral GR noise 50% at 7.09% sd; typewell GR noise 25% at 8.98% sd; vertical masking 50%, two masks up to 11.4% each; heavy augmentations active for the first 80% of training.

#### TTA

Both covered in Chapter 21.3: the **physically-correct vertical flip** (+0.13 CV) and **quarter re-anchoring** (+0.18 CV). Also noted: *"GR residual inversion helped for some models, but not ConvNeXt V2 Large. Horizontal flips were consistently harmful."*

---

### 42.2 Lightsource's pipeline — U-Net segmentation with chunked decoding

**Physical prior:** `observed_GR ≈ interp(typewell_GR, true_TVT) + noise`. Build a **geometric prior** from the known prefix and trajectory, then learn a **residual correction** to it. Two model families: 2D U-Nets doing segmentation in residual-TVT × MD space, and a 1D Squeezeformer regressing the same residual along MD.

**2D:** U-Net, encoder from scratch, 16 input channels, 1 output class. Backbones: ResNet-34, ConvNeXt-Tiny, Swin-Tiny, plus a ResNet-34 variant trained from a `geo_plane` synthetic recipe. Canvas 384 × 512, train on 256 × 256 crops; at inference either tiled 256 × 256 or single-pass full-canvas, chosen **per model on OOF**.

**16 channels:** GR–typewell mismatch, horizontal GR, residual axis, known/future masks, geometric prior features, trajectory (`z`, `dz`, `d²z`, `md`).

**Target:** mode `cum_residual` — the cumulative boundary mask of Chapter 32.2. Decode by summing the column; loss `BCE + Dice/Jaccard + row_loss(decoded) + optional smooth`.

**1D:** Squeezeformer over the well's MD axis. Features = base 1D signals **plus a GR residual profile over ±100 ft of TVT offsets (201 offsets, via a strip CNN)** — so the network sees how the GR matches the typewell *as a function of hypothesised TVT*. That is the canvas idea, compressed into an extra feature dimension for a 1D model. Target is the dense prior residual `y = TVT_true − geometric_prior`, with the known prefix masked and the loss weighted toward the unknown suffix (Huber / Gaussian-NLL style, with optional smoothness, monotonicity, and sparse-control-point auxiliaries).

**Augmentations:** an **empirical GR noise bank** (real residuals `GR − interp(typewell, TVT)` harvested from training wells) — realistic noise for free; GR dropout, gain/offset/white noise; magnitude warp; typewell swap; prior shift / slope noise / slope breaks; **coherent time stretch/warp applied jointly to GR, TVT, Z and MD** (coherent — you must warp everything together or you break the physics); mixed full-canvas vs random-crop samples; hflip, vflip; mixup; channel dropout; random known-fraction; random crop. Prefix length randomised 25–65%.

**Pretraining** ("honest synthetic"): real MD/Z and real typewells, synthetic TVT and GR. Sample a prefix length; build the geometric prior; sometimes corrupt it; add a residual (drift + spline + bumps) with `dTVT/dMD` clamped; `GR = interp(typewell, TVT) + noise_bank`.

**Post-pretraining** on a `geo_plane` recipe, where TVT is sampled from a **geological depth surface** rather than as a residual on a prior:

```
    geo_z(MD)  =  (TVT_prefix + Z)_PS  +  plane_slope·ΔMD  +  piecewise_knots(MD)
    TVT        =  geo_z − Z
```

with 4–6 piecewise-linear knots, zeroed on the known prefix and ramped into the unknown. Again this is (★) run backwards. The full synthetic stack alone, with no real fine-tuning, scored about **8.9 OOF RMSE**.

**Re-anchoring** (Chapter 33.1): decode in MD chunks of 500–950 ft, appending each chunk's predictions to the known prefix and rebuilding the prior. *"Long unknown suffixes drift if decoded in one shot."*

**Blend:** seed checkpoints averaged equally within a member; member weights from **NNLS** on OOF; each member uses its own best canvas mode and re-anchor stride. Alone ≈ 5.57 OOF.

---

### 42.3 Alijs's pipeline — particle filter plus an error-predicting LightGBM

A 64-seed particle filter matching lateral GR against the typewell, run **forward and backward**. On top, a **LightGBM residual corrector** (an L2 + L1 blend) that predicts the backbone's *error* from ~115 **self-diagnostic features**:

- particle-filter seed spread;
- stride and derivative features;
- a **Viterbi second-opinion decode**, with "typewell-rung aliasing companions" — i.e. deliberately generated alternative decodes offset by one repeat of the barcode, to detect when the filter has latched onto the wrong rung;
- forward/backward disagreement;
- disagreement with a **non-ML global stretch-squeeze DP decoder**.

Corrected predictions then pass through a projection/smoothing stage.

Every one of those features is a **disagreement measure**. The philosophy: *you cannot easily predict the truth, but you can often predict when you are wrong*, by checking whether independent methods agree.

His honest post-mortem on why it collapsed on private (9.138 vs 6.217 public):

> *"Given that every new LightGBM feature I added usually improved some wells while hurting others, the most likely explanation is that the private LB turned out to contain significantly more of that type of wells my LightGBM wasn't strong on."*

And on the method's limits: *"this competition wasn't very friendly to an LGB-style approach, as it worked at the row level and was missing the whole well-level picture."*

---

### 42.4 Arunodhayan's SDF variant — and the best ablation table in the competition

An SDF head (Chapter 32.3) on `smp.Unet` + ResNet-34, 16 input channels, initialised from Lightsource's `geo_plane` post-pretrain weights. *"Input channels are identical across modes, so encoder/decoder transfer directly; only the interpretation of the output image changes."* — a neat demonstration that a pretrained encoder–decoder is agnostic to what its output means.

Full canvas 384 × 512, no crops; single-pass inference, re-anchor 500 ft. Decode by soft expectation with τ ≈ 0.08. Stage A resumes from the `geo_plane` weights; Stage B fine-tunes on real wells for up to 50 epochs with early stopping (patience 8; stops at epoch 20–28 in practice).

**What worked:**

| change | Δ RMSE |
|---|---|
| soft decode vs zero-crossing | **−1.70** |
| equal-weight ensembling | −0.43 |
| full canvas, no crops | −0.39 (9/10 folds, p = 0.011) |
| hflip | −0.30 (4/5 folds, p = 0.028) |
| Optuna loss config | −0.16 |
| `geo_plane` pretrain | −0.15 (3/5 folds) |

**What didn't:**

| change | Δ RMSE |
|---|---|
| empirical prefix-fraction sampling | +0.357 |
| vflip p = 0.5 | +0.325 |
| mixup p = 0.5 | +0.154 |
| hard-well mining (all thresholds) | +0.15 … +0.53 |
| fitted ensemble weights | +0.079 / +0.122 |
| clip TVT to typewell range | +0.29 |
| train on longer unknown sections | +0.08 |
| loss-weight / dropout / LR sweeps | ~0 |

**The summary line every practitioner should tape to their monitor:**

> *"2 wins in ~22 five-fold experiments. Both changed what the model outputs."*

Twenty-two carefully-run experiments. Two worked. And both were changes to the **output parameterisation and decoding**, not to hyperparameters, not to architecture, not to augmentation probabilities. That is where the leverage was.

His three methodological notes are covered in Chapter 19.5: the fold-0 trap, the recipe-dependent noise floor (sd 0.103 early, 0.192 final), and selection bias across 16 seeds. Plus the nested-validation table showing fitted weights losing to equal weights (Chapter 21.2).

Final: 12 members, equal weight — 8 seeds of the locked recipe plus small config variants — **6.244 OOF RMSE**. Note also *"Unselected recipe means 7.279 → 6.910 across 16 seeds"*, i.e. the *whole recipe* improvement, measured across seeds rather than by best-of.

---

## Chapter 43. Fifth place — daimaru: synthetic-data-centric CNN

**Result:** private 5.835 for their best configuration (see Chapter 43.7 — it was not one of the two they could select), public 5.600, OOF 4.844.

The purest strategy of the five: *make the fake data so good that CV becomes trustworthy, then optimise CV.*

### 43.1 How they got there

Started on 4 July — about a month before the deadline. Early work was on characterising the distribution of `(horizontal GR − typewell GR)` and iterating on the likelihood function of a particle filter. Adding PF variations kept improving the score. But synthetic data was also improving the CNN, and generating synthetic wells on the fly *while* computing multiple particle-filter variants was too heavy. They dropped the PF entirely.

The strategic reason for the pivot:

> *"The public notebooks showed a large gap between CV and public score. By training primarily on synthetic data, I aimed to make my CV trustworthy."*

That is a subtle and excellent argument. If your model is trained mostly on synthetic wells, it has not memorised the 773 real ones, so your cross-validation on real wells is closer to an honest held-out estimate. **Synthetic pretraining is not only a data-augmentation technique; it is a validation-hygiene technique.**

### 43.2 The synthetic generator

Version 1 (Chapter 36.3): `hwGR(md) = tw(tvt(md)) + δ(tvt(md), md) + ε(md)`, with `δ` a Gaussian process over TVT × (static + slowly-swapping OU along MD) and `ε` = white noise + short-wavelength OU + long-wavelength OU.

Version 2 — the **latent geology** redesign (Chapter 36.4): generate the true unsmoothed rock column `L = L_lo + L_hi` at 0.25 ft resolution; derive the typewell as a smoothed, coarsely-sampled view `tw(tvt) = H_tw[L](tvt)`; derive the lateral as a different view of the same latent `hwGR(md) = H_hw[L](tvt(md), md) + δ + ε`.

Neither log copies fragments of real waveforms — everything is generated to be *statistically* similar.

### 43.3 The input

A 256 × 768 image (rows = TVT bins, columns = 12 ft of MD each), 14 channels:

| # | channel | content |
|---|---|---|
| 0 | typewell GR | each row's reference value, across all columns (vertical stripes) |
| 1 | hwGR | each column's 12-ft mean lateral GR, down all rows (horizontal stripes) |
| 2 | GR coverage | fraction of valid GR samples per column |
| 3 | known path | the pre-PS path as a Gaussian line; blank after PS |
| 4 | row coordinate | each row's TVT relative to the PS anchor |
| 5 | cumulative −ΔZ | where the path would sit under a flat-layer assumption |
| 6 | dip | trajectory tangent slope per column (`dz/dmd`, clipped) |
| 7 | azimuth sin | horizontal heading |
| 8 | azimuth cos | |
| 9 | typewell − hwGR misfit | ch0 − ch1 |
| 10 | context typewell GR | reference typewell of the ±800 ft neighbourhood |
| 11 | context coverage | validity mask for ch10 |
| 12 | median hwGR | column GR by median instead of mean |
| 13 | typewell − median misfit | ch0 − ch12 |

Plus a **256 × 48 appendix** concatenated on the right, drawing the typewell GR and the pre-PS lateral GR on **TVT × GR axes** (Chapter 31.5).

### 43.4 The model

```
    input 14ch × 256 × 768 (+48 appendix cols)
      ↓  ① encoder:  maxvit_tiny / effnetv2_rw_s / hrnet_w18
      ↓  ② decoder:  UNet-style
      ↓  ③ heads:    head1 = SDF (tanh × 3)
                     head2 = row-classification logits
```

Two heads on one decoder — an SDF and a per-row classification — which is a cheap form of multi-task regularisation.

**The auxiliary loss** (Chapter 31.5): the model must paint the trajectory the lateral GR would trace on the TVT × GR crossplot if the true path were followed. Worth about 0.1 ft.

### 43.5 Training and inference

25 epochs synthetic pretraining; 8 epochs real fine-tuning, **best epoch almost always epoch 2**. Augmentation on real data was just two things: shifting the PS position, and horizontal flip.

**Typewell correction at inference.** For wells sharing a typewell, build a **median residual profile from the sibling laterals** and add it to the typewell. Their reasoning: *"The typewell GR serves as the expected GR value at each TVT; because of the typewell's limited observation quality, there are places where it clearly disagrees with the horizontal GR. The other wells' horizontal GR values are used to update that expectation."* They confirmed most test wells share their typewells with the training set. (Same insight as 3rd place's sibling references — arrived at independently.)

**Adaptive canvas size** (Chapter 30.2): predict everything on the 256-row (±64 ft) canvas; flag any well whose predicted max `|relative TVT|` exceeds 55 ft; re-infer those on a 384-row (±96 ft) canvas and replace their predictions entirely.

### 43.6 Validation, and the anti-correlated leaderboard

| Sub | CV | Public | Private |
|---|---|---|---|
| v5 | 4.940 | **5.585** (their public best) | **5.993** (their private worst) |
| v6 (+ neighbour-well channel) | 4.882 | 5.631 | 5.936 |
| v8 (+ flat-drilling-well correction) | 4.87 | 5.593 | 5.940 |
| v9 | 4.870 | **5.729** (their public worst) | 5.887 |
| v10 | 4.891 | 5.618 | 5.907 |
| v11 (backbone → MaxViT-base) | **4.844** | 5.600 | **5.835** |

> *"Public and private are almost anti-correlated, while CV and private rank-correlate very well."*

This table alone justifies the entire "trust your CV" doctrine.

### 43.7 The bug

> *"My plan was to submit v11 — a 0.1 CV improvement over v5 on OOF — and select it as one of my final two. But it crashed with an error about 30 minutes before the deadline. I managed to resubmit 5 minutes before the deadline, and scoring completed 10 minutes after it. The score would have been 3rd place on private. The cause: the `img_size` argument passed to `timm` at model construction is unnecessary for MaxViT-tiny but required for MaxViT-base."*

One keyword argument, two placings. Freeze your inference pipeline early, and test the exact submission path with every backbone you intend to use.

### 43.8 The distribution-shift finding

Covered in Chapter 11.1. Their synthetic-only model scored 6.342 private — about 15th place — with **zero** real training wells. Fine-tuning improved CV and public by ~1.5 ft but private by only ~0.5 ft.

They asked the host directly whether the split was structured. The host answered that it was random and the field is a very local area of one large oil field. daimaru pushed back with evidence (the test wells appear skewed to the north-east on the task-brief map, and restricting CV to the north-eastern training wells reproduced their public score). The question was left open. Both the persistence and the humility in that exchange are worth emulating.

### 43.9 What didn't work

- training with different seeds (no gain);
- finer MD resolution.

### 43.10 The blind test

They produced eight images, a mix of real and synthetic, and challenged readers to tell which was which. *"Codex and Gemini could not tell which is which. Neither can I."* — a Turing test for synthetic data, and a good habit: if you can distinguish your fake data from real data at a glance, so can your model.


---

## Chapter 44. All five, side by side

### 44.1 The comparison matrix

| | **1st Ruby** | **2nd Bilzard** | **3rd Takoi/tereka** | **4th James et al.** | **5th daimaru** |
|---|---|---|---|---|---|
| **Core representation** | 2D canvas, 400×345 | 2D canvas, 512×336 | 1D sequences + 1D cost volume | 2D RGB render, 384×384 | 2D canvas, 256×768 |
| **Backbone** | ConvNeXt-Small U-Net | EfficientNet-B0 + FPN | dilated TCN + MaxViT1D + BiLSTM | ConvNeXt V2-Large / ResNet / Swin / Squeezeformer | MaxViT / EffNetV2 / HRNet U-Net |
| **Output head** | per-column probability map | `P(ΔTVT\|TVT)` at every anchor | 5 candidates + gate | 8 control points × 3 MTP modes | SDF + row logits |
| **Decode** | expectation | exact DP marginalisation | per-row softmax mixture | probability-weighted average | soft expectation |
| **Main loss** | smoothed cross-entropy | cross-entropy on move classes | Gaussian NLL / MSE + CE | Huber + MTP hybrid | SDF + CE + auxiliary |
| **Physics model used?** | PF as input channels | none | HMM + PF as candidates *and* features | PF (Alijs), tabular slope (James) | dropped mid-competition |
| **Synthetic data** | joint with real | 2,048 fresh wells per epoch | none reported | 500k V0 + 37k V3, staged | the centre of the strategy |
| **Equation (★) used as** | `z_diff` channel | `dip z` channel + `z_layer` internal rep | "formation rate" in the hidden state | the whole stage-1/2 pipeline | "cumulative −ΔZ" channel |
| **Ensemble** | 6 models, hand weights, routed | 15 ckpt × 8 TTA × 2 grids | 5 families × 5 splits × 5 folds | ridge over 3 pipelines | multi-backbone |
| **Distinctive idea** | XY-neighbour plane fit; trust CV over LB | anchors + exact DP; leave-largest-out test | per-row SoftMax gating; fold-safe protocol | topography-first; PhysioNet-style vision | latent-geology synthetic data |
| **CV** | 4.627 | 5.140 | 5.288 | 4.998 | 4.844 |
| **Private** | **5.639** | 5.802 | 5.836 | 5.870 | 5.835* |

### 44.2 What all five agreed on

These are the convergent findings. When five independent teams reach the same conclusion, it is probably a property of the problem, not of anyone's taste.

1. **Model the distribution; decode by expectation.** Universal. Six different implementations, one rule (Chapter 34).
2. **Equation (★) belongs in the model.** All five feed the known `Δz` in, one way or another.
3. **The 2D matching canvas.** Four of five build one explicitly; the fifth (3rd place's SDF) builds its 1D equivalent.
4. **CNNs beat transformers here.** 1st and 4th tested and said so explicitly. Small data favours strong inductive bias.
5. **Anchor everything to the last known TVT.** Universal.
6. **Trust CV, not the public leaderboard.** 1st, 2nd, 3rd and 5th all say this explicitly, and all four were right.
7. **Ensemble diversity beats individual strength.** 1st kept a model that no longer helped the single best model; 3rd kept a 9.54-RMSE component; 4th's blend survived two collapsing members.
8. **Use the pre-PS interval as a reference log, not just an anchor.** 1st (typewell calibration), 2nd (`x6`/`x7`), 3rd (self-reference in the HMM), 5th (sibling typewell correction).
9. **Exploit sibling wells sharing a typewell.** 2nd (master series), 3rd (sibling references — their single biggest private gain), 5th (median residual profile).
10. **Robust losses.** Huber, Student-t/Cauchy, or a smoothed target, everywhere. The labels are human.

### 44.3 What nobody could make work

| Idea | Who tried | Outcome |
|---|---|---|
| Transformer backbones | 1st, 4th | consistently worse than CNNs |
| Reranking / RL over a rollout bundle | 2nd | oracle gap is real but the selector always overfits |
| Explicit formation-surface modelling as a *fix* | 2nd | captures topography, cannot fix catastrophic wells |
| Removing visually bad-labelled wells | 1st | no stable gain |
| Hard-well mining | 4th (Arunodhayan) | +0.15 to +0.53 — actively harmful |
| Fitted ensemble weights (vs equal) | 4th (Arunodhayan) | worse under nested validation |
| Fancy loss-weighting schemes | 1st, 4th | no consistent improvement |
| Scaling model size / input resolution | 1st, 5th | no gain — it is a small-data problem |
| Making synthetic data maximally realistic | 4th, 26th | ambiguous at best; see Chapter 37 |
| Clipping TVT to the typewell range | 4th (Arunodhayan) | +0.29 |
| Multiple seeds (as the only change) | 5th | no gain |

### 44.4 The two-model pattern, which is the most transferable idea here

Notice how many solutions have this shape:

```
    [ Model A: physically-grounded, interpretable, mediocre alone ]
                         ↓  its prediction AND its uncertainty
    [ Model B: learned, powerful, given A's output as input ]
                         ↓
                    final answer
```

- 1st: particle filter → CNN channels
- 3rd: HMM + PF → neural nets (with the *full uncertainty profile*, not just the mean) → gate
- 4th (James): tabular slope model + its confidence intervals → vision model
- 4th (Alijs): particle filter + disagreement diagnostics → LightGBM error corrector

The two consistent refinements:

**(a) Pass the uncertainty, not just the prediction.** 3rd place's PF features include spread, skewness, IQR and five quantiles. James's green channel is a *probability*, not a value. Model B can only learn when to override Model A if it can see when Model A was unsure.

**(b) Guard against shortcut learning.** Model B will happily copy Model A and stop reading the raw data. 1st place corrupted the PF channels during training. James deliberately used a *weaker* topography model. Both are countermeasures to the same failure.

---

# PART VIII — LESSONS

---

## Chapter 45. The transferable lessons, ranked

**1. Choose the representation that makes the problem easy for the architecture you want to use.**
Turning "align two 1D signals" into "trace a path through an image" is the whole competition. Once it is an image, thirty years of computer vision — pretrained backbones, U-Nets, augmentation, TTA — becomes available. Before you tune a model, ask: *is there a coordinate system in which this problem is a solved one?*

**2. Match your decoder to your metric.**
Squared error → expectation. Not argmax, not Viterbi, not the zero crossing. This one change was worth −1.70 RMSE for one competitor. It costs nothing.

**3. Put the physics in the input, not in the model's job description.**
Equation (★) hands you half the answer. Feeding `dz` as a channel does not tell the model something it could have learned — it *removes work*, so the limited capacity and limited data go toward the part that is genuinely unknown.

**4. Build a validation scheme you trust, then actually trust it.**
Group by the right unit. Close every leakage path in a stacked pipeline. Measure your own noise floor. Use the leave-largest-contribution-out test. Then, when CV and the leaderboard disagree, investigate — and if you find nothing, go with CV. Ruby won by doing exactly that.

**5. Diversity is the currency of ensembling.**
Keep a weak model if it is wrong differently. Vary the *kind* of model, the features, the losses, the fold assignments — not just the seed.

**6. Synthetic data buys you a bigger model, and a more honest CV.**
If you can write down a generative model of your data, you can have unlimited training examples. Stage from simple to realistic to real. But do not assume realism is automatically worth its cost (Chapter 37).

**7. Predict your own errors.**
Where two independent methods disagree, something is wrong. That disagreement is a feature. Alijs built ~115 of them.

**8. Watch for shortcut learning whenever one model feeds another.**
Corrupt the helper's output during training, or use a deliberately weaker helper.

**9. Small data favours strong priors.**
CNNs over transformers. Shallow heads over deep ones. Constrained convex combinations over free stacking. Control points over free curves. Every one of these is the same trade being made the same way.

**10. Most experiments fail. Track them anyway.**
Two wins in twenty-two five-fold experiments. Both were changes to what the model outputs. The record of failures is what told him where to look.

---

## Chapter 46. A study plan

If you want to actually understand this material rather than have read about it, here is a sequence. It assumes a few weeks of evenings.

**Week 1 — Domain and data.**
Re-read Part I. Download the competition data. Plot, for one well: GR vs MD; Z vs MD; TVT vs MD; and `TVT + Z` vs MD. Verify with your own eyes that `TVT + Z` is much smoother than either component (this is equation ★, and seeing it is worth more than reading it ten times). Plot the typewell GR beside the lateral GR and try to correlate them by hand.

**Week 2 — Baselines you write yourself.**
Implement, from scratch:
1. A **constant-dip extrapolator**: estimate the recent slope of `TVT + Z` from the last 256 prefix rows, extrapolate. This is your floor. It will be bad.
2. A **naive GR matcher**: for each MD, pick the typewell TVT minimising `|GR_obs − GR_tw(TVT)|`. This will be worse, and jumpy — which teaches you *why* you need continuity constraints.
3. Combine them: minimise mismatch subject to a smoothness penalty, by dynamic programming. Now you have something reasonable.

**Week 3 — The probabilistic models.**
Implement the HMM of Chapter 25.5 and the particle filter of Chapter 26.4. Get the particle filter to ~7 RMSE. Then add, one at a time, and measure each: rare large jumps; a GR bias state; forward–backward smoothing; expectation instead of argmax decoding. Watching each of these move the number is the fastest way to internalise Part IV.

**Week 4 — The canvas.**
Build the canvas of Chapter 31.6. **Look at it.** Plot it as an image with the true path drawn on top, for twenty wells. You will immediately see the ambiguity — multiple parallel dark bands — and Chapter 34 will stop being abstract.

**Week 5 — The first neural model.**
A small U-Net (ResNet-18 encoder) on the canvas, column-softmax head, smoothed cross-entropy loss, expectation decode. Set up GroupKFold properly *before* you train anything. Establish your noise floor by running the same config with three seeds.

**Week 6 — Synthetic data.**
Write the generator of Chapter 35.3. Generate 50,000 wells. Pretrain, then fine-tune. Measure the gap. Then improve the noise model (AR residuals, a `δ_w` term) and measure again — and see for yourself whether you land in James's regime or Tucker's.

**Then:** read the five writeups again. They will read completely differently.

**Reference code:**
- 1st place training: `github.com/IAmAValidUsername/kaggle_ROGII_1st_place_solution_Ruby`
- 1st place inference: `kaggle.com/code/w5833946/submit-reproduce`
- 3rd place inference: `kaggle.com/code/tereka/rogii-exp417-gnll-five-groupkfold-submit`
- The multi-modal inversion paper: Alyaev et al., *Direct Multi-Modal Inversion of Geophysical Logs Using Deep Learning*, arXiv:2201.01871, with code at `github.com/alin256/multi-mode-prediction-with-mtp-loss`
- A domain-aware toolkit: `github.com/mycarta/rogii-geosteering-toolkit`

---

## Chapter 47. Writing the research paper

Since a paper was the stated goal, here is a structure that the material supports.

**Suggested framing.** *"Probabilistic path inference for automated geosteering: a comparative analysis of the ROGII wellbore geology prediction challenge."*

**The thesis that the evidence actually supports:** across five independent solutions using very different machinery — U-Nets, anchor CNNs, HMMs, particle filters, gradient-boosted trees, gated ensembles — performance converged on a narrow band (5.64–5.87 ft private RMSE), and the successful designs share three properties: (i) explicit representation of a multimodal posterior over stratigraphic position; (ii) expectation-based decoding matched to the squared-error metric; and (iii) incorporation of the kinematic identity ΔTVT = Δz_layer − Δz as an input feature rather than a learned relation.

**Sections:**

1. **Introduction** — geosteering, the manual workflow, the automation case.
2. **Problem formulation** — Part I and II. The identity (★) deserves its own subsection with a figure.
3. **Related work** — multi-modal log inversion (Alyaev et al.); Bayesian filtering for geosteering; sequence alignment / DTW; dense prediction architectures.
4. **The 2D alignment representation** — Part V. Argue that it is the unifying abstraction, and show that the 1D methods are its special case.
5. **A taxonomy of decoders** — argmax / Viterbi / expectation / DP marginalisation, with the decision-theoretic argument of Chapter 10.3 and the empirical evidence (2nd place's Viterbi finding, 4th place's −1.70 soft-decode measurement).
6. **Synthetic data generation** — Part VI, including the realism paradox as an open question. This is arguably the most publishable single section, because the phenomenon is real, is documented by multiple independent parties, and is not explained.
7. **Comparative analysis** — Chapter 44's matrix; the ablations that all five reported; the convergent findings.
8. **Validation under a heavy-tailed metric** — Chapters 10.2, 19.5, 19.6. The leave-largest-contribution-out test is a genuine methodological contribution worth presenting on its own.
9. **Discussion** — the public/private anti-correlation as a case study in small-sample model selection; the label-noise ceiling; what would be needed to go below 5 ft.
10. **Conclusion.**

**Two caveats to state explicitly in the paper.** First, all quantitative claims here are drawn from the authors' own competition writeups and are not independently reproduced; where a number came from "another solution branch" (as several of 3rd place's do) that should be flagged. Second, private-leaderboard differences of a tenth of a foot on 148 wells are within the noise these teams themselves measured, so the *ranking* of the top five should not be treated as a ranking of the methods.


---

# APPENDICES

---

## Appendix A — Symbols

| Symbol | Meaning |
|---|---|
| `MD` | measured depth — distance along the borehole from surface |
| `TVD` | true vertical depth — vertical distance below a datum |
| `X, Y` | map coordinates (easting, northing) |
| `Z` | vertical coordinate of the bit; known everywhere in this competition |
| `TVT` | true vertical thickness — the bit's position in the layered rock column. **The target.** |
| `TVT_input` | the given TVT values, up to the PS point |
| `GR` | gamma ray reading, API units |
| `PS` | prediction start point |
| `z_layer` | vertical position of a rock marker surface; `z_layer = TVT + z − b` |
| `Δ`, `d` | change from one row to the next |
| `f(TVT)` | the typewell's reference barcode — expected GR at stratigraphic position TVT |
| `r` | GR residual, `GR_obs − f(TVT)` |
| `δ_w` | the systematic, well-specific part of `r` |
| `ε` | the non-systematic part of `r` (autocorrelated, not white) |
| `θ` | model parameters |
| `η` | learning rate |
| `μ, σ²` | predicted mean and variance |
| `p_k(s)` | marginal probability of level `s` at column `k` |
| `α, β, γ` | HMM forward, backward, and smoothed posteriors |
| `w_i` | particle weight |
| `ESS` | effective sample size, `1 / Σ w_i²` |
| `τ` | temperature (softmax sharpness) |
| `λ` | mixing coefficient (mixup, blends) |
| `k*` | number of removals at which an improvement vanishes (leave-largest-out test) |

**Equation (★)** — the master identity: `ΔTVT = Δz_layer − Δz`.

---

## Appendix B — Glossary

### Geology and drilling

**API units** — the standard scale for gamma ray measurements.
**Azimuth** — the compass direction the borehole is heading.
**Bed** — a single layer of rock.
**Dip** — the angle at which rock layers are tilted from horizontal.
**Downdip / updip** — drilling in the direction the layers descend / rise.
**Fault** — a fracture where one block of rock has slipped past another; causes an abrupt jump in TVT.
**Formation** — a named, mappable rock unit (e.g. Buda Limestone). Formation *tops* are the marker surfaces experts pick.
**Gamma ray (GR)** — natural radioactivity of rock. High in clay-rich shale, low in clean sand, limestone, chalk.
**Geosteering** — steering a well in real time using logs to stay inside a target layer.
**Inclination / pitch** — the borehole's tilt: 0° vertical, 90° horizontal.
**Lateral** — the horizontal section of a well.
**LWD** — Logging While Drilling.
**Marker** — a recognisable surface used as a reference for measuring TVT.
**Offset well** — a nearby, already-drilled well used for reference.
**Shale / sandstone / limestone** — clay-rich / sand-derived / carbonate rock types with characteristically high / low / low gamma ray.
**Sibling wells** — horizontal wells sharing the same typewell.
**Stratigraphy** — the study of rock layering and its order.
**Structural slope / topography** — how the marker surface rises and falls across the field; `dz_layer/dMD`.
**Toe** — the far end of a lateral.
**Typewell / type log** — the vertical reference well providing the barcode `f(TVT)`.
**Well log** — a recorded curve of a measurement against depth.

### Machine learning

**Anchor** — a fixed grid position that owns a prediction (from object detection).
**Augmentation** — transforming training data in label-preserving ways to create more of it.
**Backbone** — a pretrained feature extractor reused for a new task.
**Backpropagation** — the chain-rule algorithm computing gradients through a network.
**BatchNorm / LayerNorm / GroupNorm** — normalisation layers, differing in what they average over.
**BF16** — 16-bit float with FP32's exponent range; the standard for large-model training.
**BiLSTM** — a recurrent network run in both directions.
**Cost volume** — a tensor of similarities between every query and every candidate; from stereo vision.
**Cross-attention** — attention where queries come from one sequence and keys/values from another.
**Cross-entropy** — the loss for predicting a distribution over classes.
**Dice / Jaccard loss** — overlap-based segmentation losses.
**Dilated convolution** — a convolution with gaps in its kernel; grows the receptive field exponentially with depth.
**Dropout** — randomly zeroing activations during training.
**EMA** — exponential moving average of the weights; almost always generalises better than the raw weights.
**Ensemble** — several models combined.
**Epoch** — one pass over the training data.
**FPN** — Feature Pyramid Network; fuses multi-scale backbone features.
**Fold-safe** — a validation pipeline with every leakage path closed (Chapter 19.3).
**Gaussian NLL** — a loss that makes the model predict both an answer and its own uncertainty.
**GroupKFold** — cross-validation splitting by group (here, by well).
**Huber loss** — squared error near zero, absolute error far out; outlier-robust.
**Leakage** — information from validation data reaching the training process; produces fake CV scores.
**LightGBM** — a fast gradient-boosted decision tree library.
**Manifold mixup** — mixup applied to internal representations rather than inputs.
**Mixup** — training on convex combinations of two examples and their labels.
**MTP loss** — multiple-trajectory-prediction loss; only the best of M candidates is penalised, so candidates spread across modes.
**Multimodal (posterior)** — a belief distribution with several separate peaks.
**NNLS** — non-negative least squares; used to fit ensemble weights constrained to be ≥ 0.
**OOF** — out-of-fold predictions; each made by a model that never saw that data.
**Optuna** — a hyperparameter search library.
**Quantile regression** — regression fitting a percentile instead of the mean, via the pinball loss.
**Receptive field** — how much input a given deep unit can see.
**Residual connection** — `y = x + F(x)`; makes deep networks trainable.
**RNC** — Rank-N-Contrast; a contrastive loss teaching an embedding to respect an ordering.
**SDF** — signed distance field; per-pixel signed distance to a boundary.
**SpecAugment** — masking contiguous spans of a signal during training.
**Squeezeformer** — a convolution + attention sequence model from speech recognition.
**TabPFN** — a pretrained transformer that does tabular prediction by in-context learning.
**TCN** — temporal convolutional network; a stack of (usually dilated) 1D convolutions.
**Teacher forcing** — training a sequential model using ground-truth history rather than its own predictions.
**timm** — the PyTorch Image Models library; the source of every backbone named here.
**Transfer learning** — pretrain on a large dataset, fine-tune on a small one.
**TTA** — test-time augmentation; predict on transformed copies at inference and average.
**U-Net** — an encoder–decoder with skip connections, for dense prediction.
**Weight decay** — a penalty on parameter magnitude.

### Probability and estimation

**Bayes' rule** — posterior ∝ likelihood × prior.
**Cauchy** — Student-t with 1 degree of freedom; extremely heavy-tailed, very robust as a likelihood.
**DTW** — Dynamic Time Warping; optimal alignment of two sequences allowing local stretch.
**Dynamic programming** — solving a problem by building up solutions to sub-problems on a grid.
**Emission model** — `p(observation | state)`.
**ESS** — effective sample size of a weighted particle set.
**FFBSi** — Forward Filtering Backward Simulation; a particle smoother.
**Filtering** — estimating the current state from all observations so far.
**Forward–backward** — the HMM algorithm computing smoothed posteriors.
**Genealogy tracing** — recovering particle ancestry to reconstruct full paths.
**HMM** — Hidden Markov Model; a discretised state-space model.
**Kalman filter** — the optimal filter for linear-Gaussian systems; unusable here because the posterior is multimodal.
**Likelihood** — `p(data | hypothesis)`, read as a function of the hypothesis.
**Markov property** — the future depends on the present alone.
**Ornstein–Uhlenbeck (OU)** — a mean-reverting random process; produces noise with memory.
**Particle filter / SMC** — approximating a belief distribution with a cloud of weighted samples.
**Resampling (systematic)** — a low-variance scheme for redrawing particles proportional to weight.
**Smoothing** — estimating past states using *all* observations, including future ones.
**State-space model** — a model with a hidden state, a transition model, and an emission model.
**Student-t** — a heavy-tailed alternative to the Gaussian.
**Transition model** — `p(state_t | state_{t−1})`.
**Viterbi** — the algorithm finding the single most probable path. Optimal for path accuracy; **wrong for RMSE.**

---

## Appendix C — The numbers, collected

Useful when writing up. Every figure below is as reported by the team in question.

**Final private leaderboard (top 5):** 5.639 / 5.802 / 5.836 / 5.870 / (5.835 unselected).

**Standalone component scores (CV RMSE unless stated):**

| Component | Score | Team |
|---|---|---|
| Particle filter alone | 7.4 | 1st |
| Particle filter alone | 6.58 | 3rd |
| XY-neighbour plane fit alone | 11.4 | 1st |
| 3-family HMM alone | 5.97 | 3rd |
| Adaptive 1D SDF alone | 9.54 | 3rd |
| Last-PS NN alone | 5.47 | 3rd |
| Delta NN alone | 5.77 | 3rd |
| Synthetic-only CNN, no fine-tune | 6.25 CV / **6.342 private** | 5th |
| Synthetic-only foundation model | 5.9 | 4th (James) |
| Full synthetic stack, no fine-tune | ~8.9 OOF | 4th (Lightsource) |
| Topography-only, no vision | 10.8–12.6 private | 4th (James) |

**Measured improvements:**

| Change | Δ | Team |
|---|---|---|
| soft expectation decode vs zero crossing | **−1.70** | 4th (Arunodhayan) |
| synthetic pretraining (early experiment) | −1.5 (8.2 → 6.7) | 4th (James) |
| equal-weight ensembling | −0.43 | 4th (Arunodhayan) |
| full canvas, no crops | −0.39 | 4th (Arunodhayan) |
| hflip | −0.30 | 4th (Arunodhayan) |
| XY-neighbour features (CV) | −0.30 | 1st |
| MD-phase TTA8 | −0.21 (5.624 → 5.417) | 2nd |
| quarter re-anchoring | −0.18 | 4th (James) |
| ensemble over best member | −0.17 (4.80 → 4.627) | 1st |
| Optuna loss config | −0.16 | 4th (Arunodhayan) |
| `geo_plane` pretrain | −0.15 | 4th (Arunodhayan) |
| physically-correct vertical-flip TTA | −0.13 | 4th (James) |
| sibling-distance-weighted HMM (private) | −0.126 | 3rd |
| HMM 3-family mixture | −0.079 (6.049 → 5.970) | 3rd |
| 3 → 5 GroupKFold split patterns (private) | −0.067 | 3rd |
| Gaussian NLL vs RMSE loss | −0.053 (5.418 → 5.365) | 3rd |
| initial-rate window 30 → 256 rows (private) | −0.042 | 3rd |
| TTA + re-anchor blend | −0.059 (5.201 → 5.143) | 4th (James) |
| TVT×GR auxiliary painting loss | −0.1 | 5th |
| GR-penalty loss | "small" | 1st |

**Scale figures:** 773 train wells; 3,783,989 train rows; ~200 test wells (~52 public, 148 private); 54 master typewell series; GR correlation length ≈ 18 ft; 2nd place trained 5 folds × 3 seeds in ~10 GPU-hours and inferred 200 wells in ~40 min on a T4.

---

## Appendix D — Sources

**The five solution writeups** (the primary sources for Part VII, and for every quoted number):

- Ruby, *1st Place Solution*, Kaggle, 2026 — `kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/1st-place-solution`
- Bilzard, *2nd Place Solution: AnchorCNN — Conditional Probabilistic Path Modeling*, Kaggle, 2026 — `.../writeups/2nd-place-solution-anchorcnn-conditional-probab`
- Takoi & tereka, *3rd Place Solution*, Kaggle, 2026 — `.../writeups/3rd-place-solution`
- lightsource, Arunodhayan, James Day & alijs, *4th Place Solution*, Kaggle, 2026 — `.../writeups/4th-place-solution`
- daimaru, *5th Place Solution: Synthetic-Data-Centric CNN*, Kaggle, 2026 — `.../writeups/5th-place-solution`

**Referenced literature:**

- S. Alyaev et al., *Direct Multi-Modal Inversion of Geophysical Logs Using Deep Learning*, arXiv:2201.01871 / Earth and Space Science, 2022. Code: `github.com/alin256/multi-mode-prediction-with-mtp-loss`
- J. Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection*, arXiv:1506.02640
- X. Zhou et al., *Objects as Points* (CenterNet), arXiv:1904.07850

**Domain background:**

- `github.com/mycarta/rogii-geosteering-toolkit` — a domain-aware toolkit and methodology notes for this competition
- `github.com/vamseeachanta/kaggle-rogii-2026/blob/main/docs/task-brief.md` — the extracted task brief
- The competition itself: `kaggle.com/competitions/rogii-wellbore-geology-prediction`

**A note on provenance.** Everything in Part VII, and every number in Appendix C, comes from the authors' own writeups. None of it has been independently reproduced here. Where a team reported a result from "a different solution branch" (several of 3rd place's private-LB deltas), that is noted in the text. Parts I and IV draw on standard geology and estimation-theory material; Parts III and V–VI synthesise the writeups with standard machine-learning background.

---

*End of book.*
