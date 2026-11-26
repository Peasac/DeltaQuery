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

<!-- paraphrase-sheet-v1; world seed 7, sequence seed 29, first 40 -->

## Conversation 1 (dev-0022)

1. New question: compare Jorbryn and Zeltris | disaster recovery
>

   _(currently: Jorbryn and Zeltris | disaster recovery)_
2. Re-ask: nothing changes in the question
>

   _(currently: Jorbryn and Zeltris | disaster recovery)_
3. Re-ask: nothing changes in the question
>

   _(currently: Jorbryn and Zeltris | disaster recovery)_
4. Change: add topic pricing
>

   _(currently: Jorbryn and Zeltris | disaster recovery, pricing)_
5. Change: add provider Kelmora
>

   _(currently: Jorbryn, Kelmora and Zeltris | disaster recovery, pricing)_
6. Unrelated new question: compare Lumora and Pelvane | disaster recovery
>

## Conversation 2 (dev-0024)

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
>

   _(currently: Roswick and Selnix | pricing)_
2. Unrelated new question: compare Jorbryn and Selmora | pricing
>

   _(currently: Jorbryn and Selmora | pricing)_
3. Change: add requirement RTO <= 900 s
>

   _(currently: Jorbryn and Selmora | pricing | requirements: RTO <= 900 s)_
4. Change: add provider Halwick
>

   _(currently: Halwick, Jorbryn and Selmora | pricing | requirements: RTO <= 900 s)_
5. Re-ask: nothing changes in the question
>

   _(currently: Halwick, Jorbryn and Selmora | pricing | requirements: RTO <= 900 s)_
6. Change: add requirement RPO <= 30 s
>

## Conversation 4 (test-0001)

1. New question: compare Estris and Istris | pricing
>

   _(currently: Estris and Istris | pricing)_
2. Re-ask: nothing changes in the question
>

   _(currently: Estris and Istris | pricing)_
3. Re-ask: nothing changes in the question
>

   _(currently: Estris and Istris | pricing)_
4. Change: add requirement RPO = 0 s
>

   _(currently: Estris and Istris | pricing | requirements: RPO = 0 s)_
5. Change: remove requirement RPO = 0 s
>

   _(currently: Estris and Istris | pricing)_
6. Re-ask: nothing changes in the question
>

## Conversation 5 (test-0007)

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
>

   _(currently: Arthas and Luwick | disaster recovery)_
2. Change: add requirement RTO <= 900 s
>

   _(currently: Arthas and Luwick | disaster recovery | requirements: RTO <= 900 s)_
3. Re-ask: nothing changes in the question
>

   _(currently: Arthas and Luwick | disaster recovery | requirements: RTO <= 900 s)_
4. Change: remove requirement RTO <= 900 s
>

## Conversation 7 (test-0011)

1. New question: compare Delwick and Esvell | regions
>

   _(currently: Delwick and Esvell | regions)_
2. Change: add provider Isdrick
>

   _(currently: Delwick, Esvell and Isdrick | regions)_
3. Change: add requirement RPO <= 60 s
>

   _(currently: Delwick, Esvell and Isdrick | regions | requirements: RPO <= 60 s)_
4. Change: swap provider Isdrick -> Istris
>

   _(currently: Delwick, Esvell and Istris | regions | requirements: RPO <= 60 s)_
5. Re-ask: nothing changes in the question
>

   _(currently: Delwick, Esvell and Istris | regions | requirements: RPO <= 60 s)_
6. Change: add topic backup
>

## Conversation 8 (test-0015)

1. New question: compare Pelbryn and Uldor | disaster recovery
>

   _(currently: Pelbryn and Uldor | disaster recovery)_
2. Change: add provider Esdrick
>

   _(currently: Esdrick, Pelbryn and Uldor | disaster recovery)_
3. Unrelated new question: compare Isdrick and Peltris | regions
>

   _(currently: Isdrick and Peltris | regions)_
4. Re-ask: nothing changes in the question
>

   _(currently: Isdrick and Peltris | regions)_
5. Change: add provider Esdrick
>

## Conversation 9 (test-0021)

1. New question: compare Luwick and Pelbryn | pricing
>

   _(currently: Luwick and Pelbryn | pricing)_
2. Change: add topic regions
>

   _(currently: Luwick and Pelbryn | pricing, regions)_
3. Change: add requirement RPO = 0 s
>

   _(currently: Luwick and Pelbryn | pricing, regions | requirements: RPO = 0 s)_
4. Change: remove topic pricing
>

## Conversation 10 (test-0025)

1. New question: compare Esdrick and Quinmora | pricing
>

   _(currently: Esdrick and Quinmora | pricing)_
2. Change: swap provider Esdrick -> Tavbryn
>

   _(currently: Quinmora and Tavbryn | pricing)_
3. Re-ask: nothing changes in the question
>

   _(currently: Quinmora and Tavbryn | pricing)_
4. Change: add requirement RPO <= 60 s
>

   _(currently: Quinmora and Tavbryn | pricing | requirements: RPO <= 60 s)_
5. Re-ask: nothing changes in the question
>

   _(currently: Quinmora and Tavbryn | pricing | requirements: RPO <= 60 s)_
6. Change: add topic backup
>

## Conversation 11 (test-0031)

1. New question: compare Galnix and Luwick | pricing
>

   _(currently: Galnix and Luwick | pricing)_
2. Change: add requirement RPO <= 30 s
>

   _(currently: Galnix and Luwick | pricing | requirements: RPO <= 30 s)_
3. Re-ask: nothing changes in the question
>

   _(currently: Galnix and Luwick | pricing | requirements: RPO <= 30 s)_
4. Change: swap provider Galnix -> Peltris
>

   _(currently: Luwick and Peltris | pricing | requirements: RPO <= 30 s)_
5. Re-ask: nothing changes in the question
>

   _(currently: Luwick and Peltris | pricing | requirements: RPO <= 30 s)_
6. Change: swap provider Luwick -> Tavnix
>

## Conversation 12 (test-0033)

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
>

   _(currently: Isnix and Istris | disaster recovery)_
2. Change: swap provider Istris -> Galnix
>

   _(currently: Galnix and Isnix | disaster recovery)_
3. Change: swap provider Galnix -> Quinmora
>

   _(currently: Isnix and Quinmora | disaster recovery)_
4. Re-ask: nothing changes in the question
>

   _(currently: Isnix and Quinmora | disaster recovery)_
5. Re-ask: nothing changes in the question
>

   _(currently: Isnix and Quinmora | disaster recovery)_
6. Change: add requirement RPO <= 30 s
>

## Conversation 14 (test-0037)

1. New question: compare Luwick and Tavnix | disaster recovery
>

   _(currently: Luwick and Tavnix | disaster recovery)_
2. Re-ask: nothing changes in the question
>

   _(currently: Luwick and Tavnix | disaster recovery)_
3. Re-ask: nothing changes in the question
>

   _(currently: Luwick and Tavnix | disaster recovery)_
4. Change: swap provider Luwick -> Isnix
>

   _(currently: Isnix and Tavnix | disaster recovery)_
5. Change: add topic backup
>

   _(currently: Isnix and Tavnix | backup, disaster recovery)_
6. Change: add requirement RTO <= 60 s
>

## Conversation 15 (test-0039)

1. New question: compare Isdrick and Nevmora | disaster recovery
>

   _(currently: Isdrick and Nevmora | disaster recovery)_
2. Re-ask: nothing changes in the question
>

   _(currently: Isdrick and Nevmora | disaster recovery)_
3. Unrelated new question: compare Istris and Wynnix | disaster recovery
>

   _(currently: Istris and Wynnix | disaster recovery)_
4. Change: add topic regions
>

   _(currently: Istris and Wynnix | disaster recovery, regions)_
5. Change: swap provider Istris -> Esvell
>

   _(currently: Esvell and Wynnix | disaster recovery, regions)_
6. Re-ask: nothing changes in the question
>
