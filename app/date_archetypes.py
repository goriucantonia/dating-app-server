"""The date archetype catalogue — the random draw that makes one analysis's
dates comparable to each other and different from the last analysis's.

**REVISED 2026-09-02 (owner decision): one scenario per ANALYSIS, drawn at
random, run identically against every candidate.**

What it replaces, stated plainly because it was a real design and it is being
overturned: scenarios used to be generated per candidate and anchored in that
pair's shared interests. Candidate A got a car meet, B got a bookshop, C got a
climbing gym — and then `candidate_scores` compared the three numbers as if
they had been measured the same way. They had not. A judge scoring
`conversational_flow` at a car meet and at a bookshop is running two different
experiments and reporting one league table.

The owner's requirement is a controlled comparison: **the same evening, three
different people**. Three candidates at the same cinema, disagreeing about the
same film, is a reading you can act on — the differences in the scores are
differences between the PEOPLE, because everything else was held still.

**Why a catalogue and not "ask the model for something random".** A model asked
for a first-date setting with no anchor produces a coffee shop, then a wine
bar, then a coffee shop again — its idea of a typical date is exactly what
"typical" means, and temperature does not fix that. The draw has to happen in
code, over a written-down list, or "random" is a hope. The catalogue is also
the honest place to argue about what belongs in it: every entry here is a
choice someone can see and change.

**Why the settings are interest-neutral.** The scenario is now the CONTROL
VARIABLE. Anchoring it in anyone's interests would hand one candidate a home
fixture: build the evening around A's love of vintage cars and A will look
engaged and B will look polite, and the score would be measuring the anchor
rather than the person. So the premises below are places two strangers can go
without either of them having asked for it. The cost, named: nobody gets a
date built around their own world any more. That is the price of a comparison
that means something, and the owner chose to pay it.

**Every premise contains something to have an opinion ABOUT.** That is the
whole point of the fixture — a setting where taste has nowhere to show is a
setting where three candidates produce three identical transcripts. The film,
the stall, the route, the record: each premise names the thing the two of them
will end up disagreeing over.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Archetype:
    """One entry in the catalogue.

    `key` is what gets STORED — on `analyses.scenario['archetype']` and
    therefore in every date's copy of it. It is a stable identifier, not the
    prose: the prose below can be reworded without orphaning the record of what
    a past analysis actually ran, and the no-repeat rule matches on it.

    `premise` is written as an instruction to the scenario model, not as a
    description. It says where they are, why they are there, and — the part
    that earns its place — what is sitting in front of them to have an opinion
    about.
    """

    key: str
    name: str
    premise: str


# The catalogue. Sixteen entries, all reachable by two people who have never
# met, none requiring either of them to be an enthusiast of anything.
ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        key="cinema",
        name="a small independent cinema",
        premise=(
            "A one-screen cinema showing a film neither of them has seen and "
            "neither of them chose. They have the whole walk to the seats "
            "beforehand and the whole walk out afterwards, and they did not "
            "come out of it thinking the same thing about the film."
        ),
    ),
    Archetype(
        key="night_market",
        name="a street food night market",
        premise=(
            "A covered night market with twenty stalls and no plan. Every "
            "decision is a small one made out loud — which queue, whether to "
            "share, whether the thing that looks alarming is worth ordering."
        ),
    ),
    Archetype(
        key="hill_walk",
        name="a marked trail up a hill",
        premise=(
            "A signposted walk up a hill on a bright, cold afternoon, with a "
            "fork halfway where the short way and the long way both get there. "
            "Walking side by side, so nobody is looking anybody in the eye."
        ),
    ),
    Archetype(
        key="arcade_bar",
        name="a retro arcade bar",
        premise=(
            "A basement bar full of cabinets from before either of them was "
            "born, on free play. Competitive without meaning to be, and loud "
            "enough that they have to lean in to talk."
        ),
    ),
    Archetype(
        key="museum_lates",
        name="a museum late opening",
        premise=(
            "A museum open into the evening, half-empty, drinks allowed in the "
            "galleries. Room after room of things to stand in front of and "
            "quietly disagree about."
        ),
    ),
    Archetype(
        key="cooking_class",
        name="a drop-in cooking class",
        premise=(
            "A two-hour class where strangers are paired at a bench and given "
            "one set of ingredients between them. One knife, one pan, and a "
            "recipe that has to be read by somebody."
        ),
    ),
    Archetype(
        key="record_shop",
        name="a second-hand record shop",
        premise=(
            "A cramped shop with a listening deck and crates nobody has "
            "alphabetised. Handing each other things is the entire activity, "
            "and every handover is a small claim about taste."
        ),
    ),
    Archetype(
        key="crazy_golf",
        name="a crazy golf course",
        premise=(
            "Eighteen holes of deliberately silly mini golf, keeping score on "
            "a paper card. Ridiculous enough that taking it seriously is a "
            "choice, and one of them will make it."
        ),
    ),
    Archetype(
        key="aquarium",
        name="a public aquarium",
        premise=(
            "A dim aquarium on a weekday, walking slowly past tanks. Long "
            "quiet stretches where whoever speaks first decides what the "
            "conversation is about."
        ),
    ),
    Archetype(
        key="pub_quiz",
        name="a pub quiz",
        premise=(
            "A weeknight quiz, the two of them as a team of two, six rounds "
            "and a pen. Every answer is a negotiation, and one of them has to "
            "write it down."
        ),
    ),
    Archetype(
        key="flea_market",
        name="a weekend flea market",
        premise=(
            "Trestle tables of other people's belongings under a grey sky. "
            "Picking things up, guessing prices, deciding what is treasure and "
            "what is junk — and not agreeing about which is which."
        ),
    ),
    Archetype(
        key="ice_rink",
        name="a public ice rink",
        premise=(
            "An open session at a rink where neither of them is good. Falling "
            "over in front of somebody you have just met, and whether you find "
            "that funny, is the whole evening."
        ),
    ),
    Archetype(
        key="glasshouse",
        name="a botanical glasshouse",
        premise=(
            "The heated glasshouses of a botanical garden on a wet day — warm, "
            "loud with rain on the glass, benches to sit on when the walking "
            "runs out. Slow, and nothing to hide behind."
        ),
    ),
    Archetype(
        key="karaoke_booth",
        name="a private karaoke booth",
        premise=(
            "An hour in a small booth with a tablet full of songs and nobody "
            "watching but each other. Somebody has to go first, and what they "
            "pick says something."
        ),
    ),
    Archetype(
        key="pier",
        name="a seaside pier out of season",
        premise=(
            "A long pier in the off season — wind, chips, a two-penny arcade "
            "at the end, and everything half shut. Nothing to do but walk to "
            "the end and back, talking."
        ),
    ),
    Archetype(
        key="escape_room",
        name="an escape room",
        premise=(
            "Sixty minutes locked in a themed room with a shared clock running "
            "down. Pure forced collaboration: who takes charge, who notices "
            "things, and what each of them does when they are stuck."
        ),
    ),
)

ARCHETYPES_BY_KEY: dict[str, Archetype] = {a.key: a for a in ARCHETYPES}

# How many of a user's most recent analyses' archetypes are kept out of the
# draw. This is the direct answer to "am I getting the same date every time":
# with 16 entries and the last 3 excluded, a repeat inside four consecutive
# runs is impossible rather than merely unlikely — and unlikely is what the
# old temperature-0.9 design was already claiming.
RECENT_ARCHETYPES_AVOIDED = 3


def pick_archetype(rng: random.Random, *, avoid: list[str] | None = None) -> Archetype:
    """Draw one archetype, preferring one this user has not just had.

    `avoid` is advisory ON PURPOSE. If it ever covered the whole catalogue the
    honest behaviour is to repeat a setting, not to raise — a user who has run
    seventeen analyses should get a second cinema date, not an exception in the
    pipeline. The fallback is stated here rather than left to whoever discovers
    it, and it is the only branch in this function.
    """
    avoided = set(avoid or ())
    pool = [a for a in ARCHETYPES if a.key not in avoided] or list(ARCHETYPES)
    return rng.choice(pool)
