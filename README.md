pyzx-experiments
================
For my master's thesis, I did a bunch of stuff in [pyzx](https://github.com/zxcalc/pyzx), some of which was worth it, some of which wasn't. Dumping it all in this repo for posterity. Most of the stuff I did was in the context of [There and Back Again](https://arxiv.org/abs/2003.01664).

The code in this repo was not made with the intention of being anything more than my scratch pad, so expect buggy code, Dutch-English bilingual comments, a lack of tests, lots of debug prints, choice profanity, etc.

*Please consider this repo read-only/archived.*

Contents
========
I'll list each file in the repo and give them a completely arbitrary star-rating of how interesting I deem them. For instance:  
- [My master's thesis](https://scripties.uba.uva.nl/search?id=record_55424) ★★★★★

    (Blatant self-promotion.)

General files
-------------
Code:
- `graph_helpers.py` ★★★・・

    Some methods to create and manipulate zx diagrams slightly easier than with the pyzx builtins. I use this one a lot.
- `phase_gadgets.py` ★★・・・
    
    I wanted to know if I could abuse phase gadgets a little. I couldn't, but this still contains some useful helpers to create simple phase-gadget-form zx diagrams.

Notebooks:
- `AllFeatures.ipynb` ★・・・・

    This started as "me following the tutorial `AllFeatures.ipynb` from the pyzx repo" and devolved into a scratchpad where I just built graphs. Not worth looking at.
- `playground.ipynb` ★・・・・

    Completely chaotic scratchpad.

EX Ⅰ - Does graphlike order matter?
-----------------------------------
When turning an arbitrary zx-diagram into a graphlike diagram, you can choose in what order to do this. This experiment was to see whether order matters significantly, or not. (Spoiler: it did not.)

Code:
- `graphliker.py` ★★・・・

    Whereas pyzx turns graphs into graphlikes by doing global actions, this file starts at the end and converts spiders step-by-step into a graph-like diagram.

    For some reason I implemented custom lcomp and pivot methods here, and they're terrible. (Other parts of the project sensibly use pyzx' pivot and lcomps.)

Notebooks:
-  `graphliker.ipynb` ★・・・・

    Debugging `graphliker.py`, not worth looking at in the slightest.

EX Ⅱ - Improve extraction with good lcomps (1)
----------------------------------------------
There's already simulated-annealing approaches for improving the extracted gate count by doing lcomps, but I wanted to know if I could come close manually. (Spoiler: I couldn't.) I compare various lcomp orders, and conclude that the way I do it, it just doesn't matter at all.

Code:
-  `lcomping.py` ★★・・・

    (Can I just laugh at the Dijkstra usage here?)

    While this was a terrible idea, it did have some methods `lcomp_single_vertex()` and `pivot_single_edge()` I used quite a bit in the future, so this file wasn't completely a lost cause.
- `lcomping_testing.py` ★・・・・

    Me seeing that that code wasn't worth it in the slightest no matter what I did.
- `insignificant behalve die ene grover.txt` ★・・・・

    The output of the test file.

Notebooks:
- `lcomping.ipynb` ★・・・・

    Debugging `lcomping.py`. Not worth looking at.

EX Ⅲ - Improve extraction with good lcomps (2)
-----------------------------------------------
Next, I got a different idea (or this came before and the previous came after, can't remember). What I'm doing this time is lcomp the frontier of extraction every step to minimize the amount of edges. Still not worth it.

Code:
- `extract_lcomp.py` ★★・・・

    The python implementation of what I wrote above. The way I wanted to optimize the frontier is actually a Hard problem, so, yeah...
- `extract_lcomp_testing.py`, `extract_lcomp_testing2.py` ★・・・・

    Me seeing that that code *again* wasn't worth it in the slightest no matter what I did. The outputs of the test files are commented inside the test file as well.

    I don't know why there's two test files.

EX Ⅳ - Manual circuit extraction
---------------------------------
pyzx' circuit extraction is based on a method described in There and Back Again that does not explicitly use gflow. I wanted to implement the version that *does* explicitly use gflow to help my understanding.

Code:
- `circ.py` ★★★・・

    This file is the main workhorse for manually extracting circuits. It's **unfinished**. A (Dutch) todo-list can be find on line 236, marking everything that's done (`v`), not quite properly done (`~`), not done (` `), and whatever I mean by `≈`.
- `extract_helpers.py` ★★★★・

    The algorithm in There and Back Again does not work on raw zx-diagrams, but on open graphs. This file implements this extra abstraction layer on top, handling measurements effects with lcomps and pivots as they should. This file also contains the There and Back Again triplanar gflow algorithm.

    Some methods have awkward names and do something slightly different from you may expect, so please read the docstrings.
- `test_extract_helpers.py` ★・・・・

    The single unit test file in the entire project. I can somewhat confidently state that maybe, just maybe, the code in `extract_helpers.py` isn't absolute dogshit.

Notebooks:
- `circ.ipynb` ★★・・・

    Looks like I'm testing the steps of finding gflow and handling the result.
- `circ-new.ipynb` ★・・・・

    Looks like I'm testing the steps of handling XY and YZ vertices in the extraction algorithm here.
- `phase_gadgets.ipynb` ★★・・・

    Initially for testing `phase_gadget.py`. Afterwards, also used it to test gflow code as the diagrams that file generates have very predictable gflow.

EX Ⅴ - What would become glack
------------------------------
These are experiments that eventually made me consider the concept of `glack`: what happens when a diagram does not have gflow? Most of this code is *very untested* as I nearly immediately switched back to doing math.

Code:
- `acircuit.py` ★★・・・

    If you have fewer outputs than inputs (for instance, due to measurements), you cannot have gflow. As pyzx' circuits are unitaries, and I wanted an easy way to work with non-unitaries, this file was born.

- `extract_ancilla.py` ★★★・・

    A method for extracting circuits with ancillae. This method is **incorrect**, and the one I wrote by intuition *before* I wrote the correct on in my thesis.

    In particular, while steps (2) and (*) of Theorem 4.2.3 of my thesis are (sort of?) in this code, I don't handle step (5) properly *at all*. The implementation is also kinda lazy, and instead of continuing after encountering trouble in step (*), I *restart extracting entirely*. This is *probably* correct, but no guarantees.

Notebooks:
- `challenge.ipynb` ★★★・・

    Originally a file from the pyzx repo, I tried to extract the graph with 1 extra qubit. I know I know, that's cheating, but hey. This turned into a deeper dive into the behaviour of graphs without gflow, and was the last thing I did before I switched back to math.

    This file also contains some gflow finding/printing, e.g.
    ```python
    print(find_max_delayed_flow(OpenGraph(g4), continue_on_failure=True, quiet=False))
    highlight_missing_correction_set(g4)
    ```
    prints a graph where vertices without correction set (most of them measurement artifacts or outputs, but one actual proper vertex) get drawn red. (Biggest abuse of X-spiders in pyzx history.)

    One thing to note is that while `find_max_delayed_flow` can fail and `highlight_missing_correction_set` can highlight this failure, this failure is not *minimal*. (In terms of my thesis, it's not a guaranteed certificate as the glack may be lower than what is highlighted.)