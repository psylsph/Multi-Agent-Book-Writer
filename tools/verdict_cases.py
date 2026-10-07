"""Claim/quote pairs with known answers for tools/verdict_eval.py.

The pipeline's second check asks, of a verdict that a quote SUPPORTS or
CONTRADICTS a claim: "does this quote, by itself, directly support/contradict
the claim?". The right answer is `agrees`. The costly mistake is a wrong
"yes" (a misread fact reaches the writer); a wrong "no" only drops a fact.

Each case: (kind, verdict, claim, quote, agrees)
  support     the quote states the claim: yes to "supported"
  contradict  the quote states the opposite: yes to "contradicted"
  unrelated   same topic, says nothing about the claim: no either way
  opposite    the verdict has the sign backwards: no
  overstated  the claim is stronger than the quote (all/never/always): no
Invented-free: everyday facts, none from any book.
"""

CASES = [
    # --- the quote supports the claim; verdict "supported" -> yes
    ("support", "supported", "Triage nurses sort patients by urgency on arrival.",
     "Nurses use a triage scale to sort patients by urgency as soon as they arrive.", True),
    ("support", "supported", "The River Thames flows through London.",
     "The River Thames flows through central London on its way to the North Sea.", True),
    ("support", "supported", "Pampas grass is hardy in southern England.",
     "Pampas grass is fully hardy across most of the UK and survives winter frosts in southern England.", True),
    ("support", "supported", "Mount Snowdon is 1,085 metres high.",
     "Snowdon, at 1,085 metres, is the highest mountain in Wales.", True),
    ("support", "supported",
     "Private physiotherapists in the UK can be booked without a GP referral.",
     "You can usually book a private physiotherapist directly without a GP referral.", True),
    ("support", "supported", "Bats are not blind.",
     "Bats are not blind; all species can see, and many also use echolocation.", True),
    ("support", "supported", "Ferries to the Isle of Wight sail from Portsmouth.",
     "Wightlink car ferries sail from Portsmouth to Fishbourne on the Isle of Wight.", True),

    # --- the quote states the opposite; verdict "contradicted" -> yes
    ("contradict", "contradicted", "Pampas grass cannot survive English winters.",
     "Pampas grass is fully hardy in the UK and survives winters well.", True),
    ("contradict", "contradicted", "Village halls never host yoga classes.",
     "Many village halls across the UK host weekly yoga classes.", True),
    ("contradict", "contradicted", "A barn conversion needs no planning permission.",
     "Converting a barn to a home normally requires planning permission or prior approval.", True),
    ("contradict", "contradicted", "Ravens are smaller than crows.",
     "The common raven is considerably larger than the carrion crow.", True),
    ("contradict", "contradicted", "The Thames Barrier was completed in 1982.",
     "The Thames Barrier was completed in 1984 and became operational that year.", True),
    ("contradict", "contradicted", "Tomatoes are poisonous to eat.",
     "Tomatoes are not poisonous; the fruit is safe to eat, though the leaves are toxic.", True),
    ("contradict", "contradicted", "Mount Snowdon is 1,200 metres high.",
     "Snowdon, at 1,085 metres, is the highest mountain in Wales.", True),
    ("contradict", "contradicted", "Pampas grass always dies in frost.",
     "Pampas grass may suffer in severe frost but usually recovers in spring.", True),

    # --- same topic, says nothing about the claim -> no
    ("unrelated", "supported", "Triage nurses sort patients by urgency on arrival.",
     "Nurses on hospital wards typically wear blue or navy uniforms.", False),
    ("unrelated", "supported", "Pampas grass is hardy in southern England.",
     "Pampas grass produces tall silvery plumes in late summer.", False),
    ("unrelated", "supported", "The River Thames flows through London.",
     "The Severn is the longest river in Great Britain.", False),
    ("unrelated", "supported", "Ferries to the Isle of Wight sail from Portsmouth.",
     "The Isle of Wight is known for its sandy beaches and dinosaur fossils.", False),
    ("unrelated", "supported", "Heathrow is the busiest airport in the UK.",
     "Gatwick is the busiest single-runway airport in the world.", False),
    ("unrelated", "contradicted", "Village halls never host yoga classes.",
     "Village halls are often hired for weddings, markets and children's parties.", False),
    ("unrelated", "contradicted", "A barn conversion needs no planning permission.",
     "Many converted barns have exposed oak beams and large glass frontages.", False),
    ("unrelated", "contradicted", "Ravens are smaller than crows.",
     "Ravens are often seen at the Tower of London, where six are kept.", False),

    # --- the verdict has the sign backwards -> no
    ("opposite", "supported", "Pampas grass is hardy in southern England.",
     "Pampas grass is not hardy in the UK and usually dies over winter.", False),
    ("opposite", "supported", "The Thames Barrier was completed in 1982.",
     "The Thames Barrier was completed in 1984 and became operational that year.", False),
    ("opposite", "supported", "Tomatoes are poisonous to eat.",
     "Tomatoes are not poisonous; the fruit is safe to eat, though the leaves are toxic.", False),
    ("opposite", "supported", "Mount Snowdon is 1,200 metres high.",
     "Snowdon, at 1,085 metres, is the highest mountain in Wales.", False),
    ("opposite", "supported", "Bats are blind.",
     "Bats are not blind; all species can see, and many also use echolocation.", False),
    ("opposite", "contradicted", "Triage nurses sort patients by urgency on arrival.",
     "Nurses use a triage scale to sort patients by urgency as soon as they arrive.", False),
    ("opposite", "contradicted", "The River Thames flows through London.",
     "The River Thames flows through central London on its way to the North Sea.", False),
    ("opposite", "contradicted", "Mount Snowdon is 1,085 metres high.",
     "Snowdon, at 1,085 metres, is the highest mountain in Wales.", False),

    # --- the claim is stronger than the quote -> no
    ("overstated", "supported", "All UK clinics offer direct booking.",
     "Many private clinics let you book directly.", False),
    ("overstated", "supported", "Pampas grass never suffers frost damage.",
     "Pampas grass may suffer in severe frost but usually recovers in spring.", False),
    ("overstated", "supported", "Every village hall in the UK hosts yoga classes.",
     "Many village halls across the UK host weekly yoga classes.", False),
]

KINDS = ("support", "contradict", "unrelated", "opposite", "overstated")
