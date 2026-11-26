# Paraphrase sheet

Write each message the way *you* would type it to a chat assistant. One message per `>` line.

- Don't copy the description's wording. Vary it: casual, terse, a typo here and there is fine.
- Say only what the line asks for. Each message is a reply in an ongoing chat about the
  previous state, so "drop the second one" style references are fine only if unambiguous.
- Provider names must appear as written (the compiler needs to recognise them; typos in
  names are a different test).
- "Re-ask" lines: ask the same question again without changing anything (e.g. you heard the
  documents changed). Don't restate the whole question unless you would naturally.
- Please don't look at `deltaquery/domain.py` or the grounding code before or while writing.
- Leave a `>` line empty to skip it; a sequence with any empty line is left out.

<!-- filled by ollama/qwen3.5:9b (temperature 0, seed 0) from the meaning descriptions only; conversations whose messages contradict the gold meaning are excluded (answers removed, reason noted) -->
<!-- paraphrase-sheet-v1; world seed 7, sequence seed 29, first 40 -->

## Conversation 1 (dev-0022)

1. New question: compare Jorbryn and Zeltris | disaster recovery
> hey, can you compare Jorbryn and Zeltris for disaster recovery?

   _(currently: Jorbryn and Zeltris | disaster recovery)_
2. Re-ask: nothing changes in the question
> still need that comparison between Jorbryn and Zeltris on disaster recovery

   _(currently: Jorbryn and Zeltris | disaster recovery)_
3. Re-ask: nothing changes in the question
> wait, let's look at Jorbryn vs Zeltris again for disaster recovery

   _(currently: Jorbryn and Zeltris | disaster recovery)_
4. Change: add topic pricing
> okay now add pricing to the Jorbryn vs Zeltris disaster recovery check

   _(currently: Jorbryn and Zeltris | disaster recovery, pricing)_
5. Change: add provider Kelmora
> what about Kelmora too? compare Jorbryn, Zeltris, and Kelmora on disaster recovery

   _(currently: Jorbryn, Kelmora and Zeltris | disaster recovery, pricing)_
6. Unrelated new question: compare Lumora and Pelvane | disaster recovery
> switch topics: compare Lumora and Pelvane for disaster recovery

## Conversation 2 (dev-0024)
<!-- EXCLUDED: line 6 names Selvell and Ultris only; current question also has Rostris. Generated: hey, can you compare Selvell and Ultris for disaster recovery? | still want to see Selvell vs Ultris on disaster recovery | same thing please, Selvell and Ultris disaster recovery comparison | oops wrong name, swap Ultris for Rostris now, so Selvell vs Rostris for disaster recovery | wait, add Ultris back in too, so compare Selvell, Rostris, and Ultris for disaster recovery | back to the original request, just Selvell and Ultris on disaster recovery again -->

1. New question: compare Selvell and Ultris | disaster recovery
>

   _(currently: Selvell and Ultris | disaster recovery)_
2. Re-ask: nothing changes in the question
>

   _(currently: Selvell and Ultris | disaster recovery)_
3. Re-ask: nothing changes in the question
>

   _(currently: Selvell and Ultris | disaster recovery)_
4. Change: swap provider Ultris -> Rostris
>

   _(currently: Rostris and Selvell | disaster recovery)_
5. Change: add provider Ultris
>

   _(currently: Rostris, Selvell and Ultris | disaster recovery)_
6. Re-ask: nothing changes in the question
>

## Conversation 3 (dev-0038)

1. New question: compare Roswick and Selnix | pricing
> hey, can you compare Roswick and Selnix pricing?

   _(currently: Roswick and Selnix | pricing)_
2. Unrelated new question: compare Jorbryn and Selmora | pricing
> switching gears, what about Jorbryn vs Selmora for price?

   _(currently: Jorbryn and Selmora | pricing)_
3. Change: add requirement RTO <= 900 s
> oops, add a requirement: RTO needs to be <= 900 s

   _(currently: Jorbryn and Selmora | pricing | requirements: RTO <= 900 s)_
4. Change: add provider Halwick
> also include Halwick in the comparison now

   _(currently: Halwick, Jorbryn and Selmora | pricing | requirements: RTO <= 900 s)_
5. Re-ask: nothing changes in the question
> wait, let's ask that again without changing anything

   _(currently: Halwick, Jorbryn and Selmora | pricing | requirements: RTO <= 900 s)_
6. Change: add requirement RPO <= 30 s
> one more thing, add RPO <= 30 s to the list

## Conversation 4 (test-0001)

1. New question: compare Estris and Istris | pricing
> hey, can you compare Estris and Istris pricing?

   _(currently: Estris and Istris | pricing)_
2. Re-ask: nothing changes in the question
> still wanna see Estris vs Istris prices

   _(currently: Estris and Istris | pricing)_
3. Re-ask: nothing changes in the question
> same thing again: Estris and Istris pricing comparison

   _(currently: Estris and Istris | pricing)_
4. Change: add requirement RPO = 0 s
> ok now add RPO = 0 s to the Estris vs Istris price check

   _(currently: Estris and Istris | pricing | requirements: RPO = 0 s)_
5. Change: remove requirement RPO = 0 s
> oops, take that RPO = 0 s requirement away for Estris and Istris pricing

   _(currently: Estris and Istris | pricing)_
6. Re-ask: nothing changes in the question
> back to just comparing Estris and Istris prices

## Conversation 5 (test-0007)
<!-- EXCLUDED: lines 5-6 name two of the three current providers. Generated: hey, can you compare pricing for Estris and Isdrick? | still wanna see Estris vs Isdrick prices? | oops, forgot to add Quinmora to the list, let's include that one too | wait, swap Quinmora out and put Esdrick in instead | so just Esdrick and Isdrick pricing again? | yeah, still need the price comparison for Estris and Isdrick -->

1. New question: compare Estris and Isdrick | pricing
>

   _(currently: Estris and Isdrick | pricing)_
2. Re-ask: nothing changes in the question
>

   _(currently: Estris and Isdrick | pricing)_
3. Change: add provider Quinmora
>

   _(currently: Estris, Isdrick and Quinmora | pricing)_
4. Change: swap provider Quinmora -> Esdrick
>

   _(currently: Esdrick, Estris and Isdrick | pricing)_
5. Re-ask: nothing changes in the question
>

   _(currently: Esdrick, Estris and Isdrick | pricing)_
6. Re-ask: nothing changes in the question
>

## Conversation 6 (test-0009)

1. New question: compare Arthas and Luwick | disaster recovery
> hey, can you compare Arthas and Luwick for disaster recovery?

   _(currently: Arthas and Luwick | disaster recovery)_
2. Change: add requirement RTO <= 900 s
> also need RTO to be at most 900 seconds

   _(currently: Arthas and Luwick | disaster recovery | requirements: RTO <= 900 s)_
3. Re-ask: nothing changes in the question
> wait, just asking again about Arthas vs Luwick for DR

   _(currently: Arthas and Luwick | disaster recovery | requirements: RTO <= 900 s)_
4. Change: remove requirement RTO <= 900 s
> ok so forget the RTO limit, just compare Arthas and Luwick for disaster recovery

## Conversation 7 (test-0011)

1. New question: compare Delwick and Esvell | regions
> hey, can you compare Delwick and Esvell? i need to see which one has more regions available.

   _(currently: Delwick and Esvell | regions)_
2. Change: add provider Isdrick
> actually, let's also throw Isdrick into the mix for this comparison.

   _(currently: Delwick, Esvell and Isdrick | regions)_
3. Change: add requirement RPO <= 60 s
> oh wait, we need a stricter requirement now: make sure the RPO is at most 60 seconds.

   _(currently: Delwick, Esvell and Isdrick | regions | requirements: RPO <= 60 s)_
4. Change: swap provider Isdrick -> Istris
> oops, my bad. swap Isdrick out and put Istris in instead of it.

   _(currently: Delwick, Esvell and Istris | regions | requirements: RPO <= 60 s)_
5. Re-ask: nothing changes in the question
> never mind, just keep everything exactly as it is right now.

   _(currently: Delwick, Esvell and Istris | regions | requirements: RPO <= 60 s)_
6. Change: add topic backup
> one more thing, can you also factor backup capabilities into this comparison?

## Conversation 8 (test-0015)

1. New question: compare Pelbryn and Uldor | disaster recovery
> hey, can you compare Pelbryn and Uldor for disaster recovery?

   _(currently: Pelbryn and Uldor | disaster recovery)_
2. Change: add provider Esdrick
> actually, let's also include Esdrick in that comparison

   _(currently: Esdrick, Pelbryn and Uldor | disaster recovery)_
3. Unrelated new question: compare Isdrick and Peltris | regions
> oops wrong topic, how about comparing Isdrick and Peltris on regions instead?

   _(currently: Isdrick and Peltris | regions)_
4. Re-ask: nothing changes in the question
> wait, just asking again to compare Isdrick and Peltris regarding regions

   _(currently: Isdrick and Peltris | regions)_
5. Change: add provider Esdrick
> and please add Esdrick back into the list for this region comparison

## Conversation 9 (test-0021)

1. New question: compare Luwick and Pelbryn | pricing
> hey, can you compare Luwick and Pelbryn for me? i need to see their pricing.

   _(currently: Luwick and Pelbryn | pricing)_
2. Change: add topic regions
> also, could you include info on which regions they support?

   _(currently: Luwick and Pelbryn | pricing, regions)_
3. Change: add requirement RPO = 0 s
> wait, actually i need the RPO to be exactly 0 seconds. please update that.

   _(currently: Luwick and Pelbryn | pricing, regions | requirements: RPO = 0 s)_
4. Change: remove topic pricing
> oops, ignore the pricing part then. just focus on the regions and the zero RPO requirement.

## Conversation 10 (test-0025)

1. New question: compare Esdrick and Quinmora | pricing
> hey, can you compare Esdrick and Quinmora for pricing?

   _(currently: Esdrick and Quinmora | pricing)_
2. Change: swap provider Esdrick -> Tavbryn
> oops, actually swap Esdrick out for Tavbryn instead

   _(currently: Quinmora and Tavbryn | pricing)_
3. Re-ask: nothing changes in the question
> wait, just ask the same thing again without changing anything

   _(currently: Quinmora and Tavbryn | pricing)_
4. Change: add requirement RPO <= 60 s
> add a requirement that RPO needs to be <= 60 s

   _(currently: Quinmora and Tavbryn | pricing | requirements: RPO <= 60 s)_
5. Re-ask: nothing changes in the question
> ask the current question again with no changes

   _(currently: Quinmora and Tavbryn | pricing | requirements: RPO <= 60 s)_
6. Change: add topic backup
> also include backup in the comparison

## Conversation 11 (test-0031)

1. New question: compare Galnix and Luwick | pricing
> hey, can you compare pricing for Galnix and Luwick?

   _(currently: Galnix and Luwick | pricing)_
2. Change: add requirement RPO <= 30 s
> also need RPO to be at most 30 seconds

   _(currently: Galnix and Luwick | pricing | requirements: RPO <= 30 s)_
3. Re-ask: nothing changes in the question
> still want the price comparison for those two

   _(currently: Galnix and Luwick | pricing | requirements: RPO <= 30 s)_
4. Change: swap provider Galnix -> Peltris
> now swap Galnix for Peltris and keep asking about pricing

   _(currently: Luwick and Peltris | pricing | requirements: RPO <= 30 s)_
5. Re-ask: nothing changes in the question
> just comparing Peltris and Luwick prices again

   _(currently: Luwick and Peltris | pricing | requirements: RPO <= 30 s)_
6. Change: swap provider Luwick -> Tavnix
> finally, switch Luwick to Tavnix and show me the pricing for Peltris vs Tavnix

## Conversation 12 (test-0033)
<!-- EXCLUDED: lines 3-4 return to Delmora/Tavnix; current question is Peltris/Wynnix. Generated: hey, can you check the prices for Delmora vs Tavnix? | switching gears, what about Peltris and Wynnix pricing? | wait, just asking again about Delmora and Tavnix costs | oh right, also need to see disaster recovery info for those two | new topic: Estris vs Nevmora prices please | how does Arthas compare to Delmora on disaster recovery? -->

1. New question: compare Delmora and Tavnix | pricing
>

   _(currently: Delmora and Tavnix | pricing)_
2. Unrelated new question: compare Peltris and Wynnix | pricing
>

   _(currently: Peltris and Wynnix | pricing)_
3. Re-ask: nothing changes in the question
>

   _(currently: Peltris and Wynnix | pricing)_
4. Change: add topic disaster recovery
>

   _(currently: Peltris and Wynnix | disaster recovery, pricing)_
5. Unrelated new question: compare Estris and Nevmora | pricing
>

   _(currently: Estris and Nevmora | pricing)_
6. Unrelated new question: compare Arthas and Delmora | disaster recovery
>

## Conversation 13 (test-0035)

1. New question: compare Isnix and Istris | disaster recovery
> hey, can you compare Isnix and Istris for disaster recovery?

   _(currently: Isnix and Istris | disaster recovery)_
2. Change: swap provider Istris -> Galnix
> actually, swap out Istris and use Galnix instead

   _(currently: Galnix and Isnix | disaster recovery)_
3. Change: swap provider Galnix -> Quinmora
> oops, change that to Quinmora now

   _(currently: Isnix and Quinmora | disaster recovery)_
4. Re-ask: nothing changes in the question
> wait, just ask the same thing again without changing anything

   _(currently: Isnix and Quinmora | disaster recovery)_
5. Re-ask: nothing changes in the question
> ask it one more time exactly like before

   _(currently: Isnix and Quinmora | disaster recovery)_
6. Change: add requirement RPO <= 30 s
> also add a requirement where RPO has to be at most 30 seconds

## Conversation 14 (test-0037)

1. New question: compare Luwick and Tavnix | disaster recovery
> hey, can you compare Luwick and Tavnix for disaster recovery?

   _(currently: Luwick and Tavnix | disaster recovery)_
2. Re-ask: nothing changes in the question
> still comparing Luwick and Tavnix on disaster recovery

   _(currently: Luwick and Tavnix | disaster recovery)_
3. Re-ask: nothing changes in the question
> Luwick vs Tavnix disaster recovery comparison please

   _(currently: Luwick and Tavnix | disaster recovery)_
4. Change: swap provider Luwick -> Isnix
> now swap Luwick for Isnix but keep the disaster recovery topic

   _(currently: Isnix and Tavnix | disaster recovery)_
5. Change: add topic backup
> also add backup to the comparison with Isnix and Tavnix

   _(currently: Isnix and Tavnix | backup, disaster recovery)_
6. Change: add requirement RTO <= 60 s
> and make sure the RTO is at most 60 seconds for Isnix and Tavnix

## Conversation 15 (test-0039)

1. New question: compare Isdrick and Nevmora | disaster recovery
> hey, can you compare Isdrick and Nevmora for disaster recovery?

   _(currently: Isdrick and Nevmora | disaster recovery)_
2. Re-ask: nothing changes in the question
> still want to see how Isdrick stacks up against Nevmora on disaster recovery

   _(currently: Isdrick and Nevmora | disaster recovery)_
3. Unrelated new question: compare Istris and Wynnix | disaster recovery
> switching gears, what about Istris vs Wynnix for disaster recovery?

   _(currently: Istris and Wynnix | disaster recovery)_
4. Change: add topic regions
> oh right, let's also include regions in that comparison

   _(currently: Istris and Wynnix | disaster recovery, regions)_
5. Change: swap provider Istris -> Esvell
> oops, swap Istris out for Esvell, so compare Esvell and Wynnix on disaster recovery with regions included

   _(currently: Esvell and Wynnix | disaster recovery, regions)_
6. Re-ask: nothing changes in the question
> wait, just to confirm, compare Esvell and Wynnix for disaster recovery including regions?
