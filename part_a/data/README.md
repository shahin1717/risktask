# *Azərbaycanca* &mdash; Azerbaijani (`az`)

This datasheet is for cv-corpus-25.0-2026-03-09 of the Mozilla Common Voice *Scripted Speech* dataset for Azerbaijani [Azərbaycanca - `az`]. The dataset contains 1009 clips representing 1.52 hours of recorded speech (0.65 hours validated) from 45 speakers, recorded from a text corpus of 95,023 sentences.

## Language

### Accents

| Code | Accent | Clips | Speakers |
|---|---|---|---|
| - |  | 45 (4.5%) | 2 (4.4%) |

## Demographic information

The dataset includes the following self-declared age and gender distributions. A coverage summary is shown below each table.

### Gender

Self-declared gender information. The table shows clip and speaker counts with percentages. Speakers who did not declare a gender are listed as Unspecified. A dash (-) indicates zero.

| Code | Gender | Clips | Speakers |
|---|---|---|---|
| male_masculine | Male, masculine | 168 (16.7%) | 8 (17.8%) |
| female_feminine | Female, feminine | 10 (1.0%) | 1 (2.2%) |
| transgender | Transgender | - | - |
| non-binary | Non-binary | - | - |
| do_not_wish_to_say | Prefer not to say | - | - |
| - | Unspecified | 831 (82.4%) | 38 (84.4%) |

*Gender declared: 178 of 1,009 clips (17.6%), 7 of 45 speakers (15.6%)*

### Age

Self-declared age information. The table shows clip and speaker counts with percentages. Speakers who did not declare an age are listed as Unspecified. A dash (-) indicates zero.

| Code | Age | Clips | Speakers |
|---|---|---|---|
| teens | Teens | 15 (1.5%) | 2 (4.4%) |
| twenties | Twenties | 176 (17.4%) | 9 (20.0%) |
| thirties | Thirties | 613 (60.8%) | 5 (11.1%) |
| fourties | Fourties | 8 (0.8%) | 2 (4.4%) |
| fifties | Fifties | - | - |
| sixties | Sixties | - | - |
| seventies | Seventies | - | - |
| eighties | Eighties | - | - |
| nineties | Nineties | - | - |
| - | Unspecified | 197 (19.5%) | 30 (66.7%) |

*Age declared: 812 of 1,009 clips (80.5%), 15 of 45 speakers (33.3%)*

## Data splits for modelling

**Clip buckets**

| Bucket | Clips |
|---|---|
| Validated | 434 (43.0%) |
| Invalidated | 54 (5.4%) |
| Other | 521 (51.6%) |

**Training splits**

| Split | Clips |
|---|---|
| Train | 215 (49.5%) |
| Dev | 93 (21.4%) |
| Test | 126 (29.0%) |

*Training split coverage: 434 of 434 validated clips (100.0%)*

The dataset contains 434 validated, 54 invalidated, and 521 unresolved clips. The average clip duration is 5.452 seconds.

## Text corpus

**Validated sentences:** 93,161

| Category | Count |
|---|---|
| Unvalidated sentences | 1,862 |
| Pending sentences | 1,852 |
| Rejected sentences | 10 |
| Reported sentences | 5 |

The corpus contains 95,023 sentences: 93,161 validated and 1,862 unvalidated (1,852 pending review, 10 rejected), with 5 reported for review.

### Sample

There follows a randomly selected sample of five sentences from the corpus.

1. *Han ona qulluq etməyə başladıqdan qısa müddət sonra öldü.*
2. *Donuzlav yarımadanın ətrafında yerləşən bir neçə göldən biridir.*
3. *Şəhər xətlərin əksəriyyətini tikib və bu şirkətlərə icarəyə vermişdi.*
4. *Sonra əhalisinin çoxu türklərdən ibarət olan Qıpçaq vilayətinə getdilər.*
5. *Özü də həmin məktəbdə dərs demişdi.*

### Sources

| Source | Sentences |
|---|---|
| wiki | 93,142 (100.0%) |
| Other | 19 (0.0%) |

### Fields

#### Clips

Each row of a `tsv` file represents a single audio clip, and contains the following information:

- `client_id` - hashed UUID of a given user
- `path` - relative path of the audio file
- `text` - supposed transcription of the audio
- `up_votes` - number of people who said audio matches the text
- `down_votes` - number of people who said audio does not match text
- `age` - age of the speaker[^1]
- `gender` - gender of the speaker[^1]
- `accents` - accents of the speaker[^1]
- `variant` - variant of the language[^1]
- `segment` - if sentence belongs to a custom dataset segment, it will be listed here
- `prompt_upvotes` - number of upvotes the sentence prompt received
- `prompt_reports` - number of reports the sentence prompt received
- `is_edited` - whether the clip's transcription has been edited

[^1]: For a full list of age, gender, and accent options, see the [demographics spec](https://github.com/common-voice/common-voice/blob/main/web/src/stores/demographics.ts). These will only be reported if the speaker opted in to provide that information.

#### `validated_sentences.tsv`

The `validated_sentences.tsv` file contains one row per validated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `is_used` - whether the sentence is still in circulation for recording
- `clips_count` - number of clips recorded for this sentence

#### `unvalidated_sentences.tsv`

The `unvalidated_sentences.tsv` file contains one row per unvalidated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `up_votes` - number of upvotes the sentence received
- `down_votes` - number of downvotes the sentence received
- `status` - current status of the sentence (`pending` or `rejected`)

## Get involved

### Community links

- [Common Voice translators on Pontoon](https://pontoon.mozilla.org/az/common-voice/contributors/)
- [Common Voice Communities](https://github.com/common-voice/common-voice/blob/main/docs/COMMUNITIES.md)

### Discussions

- [Common Voice on Matrix](https://chat.mozilla.org/#/room/#common-voice:mozilla.org)
- [Common Voice on Discourse](https://discourse.mozilla.org/t/about-common-voice-readme-first/17218)
- [Common Voice on Discord](https://discord.gg/9QTj9zwn)
- [Common Voice on Telegram](https://t.me/mozilla_common_voice)

### Contribute

- [Speak](https://commonvoice.mozilla.org/az/speak)
- [Write](https://commonvoice.mozilla.org/az/write)
- [Listen](https://commonvoice.mozilla.org/az/listen)
- [Review](https://commonvoice.mozilla.org/az/review)

## Licence

This dataset is released under the [Creative Commons Zero (CC-0)](https://creativecommons.org/public-domain/cc0/) licence. By downloading this data you agree to not determine the identity of speakers in the dataset.
