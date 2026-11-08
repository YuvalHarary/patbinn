# Evaluated taxonomy

The 58 classes the classifier resolves, across 8 families and 542 NCBI isolates, with the accessions behind each class.

Per-isolate detail (sequence length, defline, fold) is in [GENOME_INVENTORY.txt](GENOME_INVENTORY.txt). Regenerate this file with `python3 scripts/make_taxonomy.py`.


## What "variant" means

The label is not a single taxonomic rank. It is the level at which a family is clinically distinguished, which differs between families, so a per-class number should be read against the rank below rather than as a uniform one.

| Family | Classes | Isolates | What one class is |
|---|---|---|---|
| Coronaviridae | 7 | 70 | Distinct human coronaviruses. Six are separate ICTV species; SARS-CoV-1 and SARS-CoV-2 are two viruses within the single species *Severe acute respiratory syndrome-related coronavirus*. |
| Flaviviridae | 8 | 80 | Separate species. The four dengue classes are the four dengue serotypes, which ICTV also treats as four species. |
| Filoviridae | 8 | 64 | Mostly separate species: the five ebolaviruses and Lloviu cuevavirus. Marburg and Ravn are two viruses within the single species *Marburg marburgvirus*. |
| Hepadnaviridae | 8 | 71 | Genotypes A to H of one species, hepatitis B virus. This is the finest-grained split in the taxonomy, and the only family whose classes are all one species. |
| Retroviridae | 3 | 21 | Two HIV-1 subtypes (clades B and C of one species) plus HIV-2, a separate species. |
| Orthoherpesviridae | 8 | 76 | Separate species, including HHV-6A and HHV-6B, which ICTV separated into two species in 2012. |
| Picornaviridae | 8 | 80 | Mixed. Poliovirus 1 to 3 are three serotypes of *Enterovirus C*; Coxsackievirus A16 and B3 belong to *Enterovirus A* and *Enterovirus B*; rhinovirus A, B and C are three species. |
| Paramyxoviridae [^1] | 8 | 80 | Separate species, except RSV-A and RSV-B, which are two subgroups of human respiratory syncytial virus. |

[^1]: **Paramyxoviridae.** This branch spans two families as currently classified. RSV-A, RSV-B and human metapneumovirus were moved to *Pneumoviridae* by ICTV in 2016; the remaining five classes are *Paramyxoviridae*. The branch is a grouping of the classifier, named for the family most of its classes belong to, and is called a family throughout the code and the manifests.


## Classes and accessions


### Coronaviridae

7 classes, 70 isolates.

| Class | Isolates | Accessions |
|---|---|---|
| SARS-CoV-2 | 10 | `NC_045512.2`, `OZ492125.1`, `OZ492124.1`, `OZ492123.1`, `OZ492122.1`, `OZ492121.1`, `OZ492120.1`, `OZ492119.1`, `OZ492118.1`, `OZ492117.1` |
| SARS-CoV-1 | 10 | `NC_004718.3`, `JX163928.1`, `JX163927.1`, `JX163926.1`, `JX163925.1`, `JX163924.1`, `JX163923.1`, `AY545919.1`, `AY545918.1`, `AY545917.1` |
| MERS-CoV | 10 | `NC_019843.3`, `PZ439299.1`, `PZ397813.1`, `PX660034.1`, `PV239413.1`, `PV239412.1`, `PV239411.1`, `PV239410.1`, `PV239409.1`, `PV239408.1` |
| HCoV-229E | 10 | `NC_002645.1`, `PZ496046.1`, `PX404513.1`, `LC654445.1`, `OZ470519.1`, `OZ470518.1`, `LC922646.1`, `PZ086308.1`, `PZ086307.1`, `PZ086306.1` |
| HCoV-NL63 | 10 | `NC_005831.2`, `LC720428.1`, `LC687402.1`, `LC687394.1`, `LC895504.1`, `LC922677.1`, `LC922676.1`, `LC922664.1`, `LC922663.1`, `PX640308.1` |
| HCoV-OC43 | 10 | `NC_006213.1`, `PZ496039.1`, `PV798440.1`, `PV798439.1`, `PV798438.1`, `PV798437.1`, `PV798436.1`, `PV798435.1`, `PV798434.1`, `PV798433.1` |
| HCoV-HKU1 | 10 | `NC_006577.2`, `PV619111.1`, `PV619110.1`, `PZ436647.1`, `PZ436646.1`, `PZ436645.1`, `PZ436644.1`, `PZ436643.1`, `PZ436642.1`, `PZ436641.1` |

### Flaviviridae

8 classes, 80 isolates.

| Class | Isolates | Accessions |
|---|---|---|
| Dengue-1 | 10 | `NC_001477.1`, `PZ602941.1`, `PZ602905.1`, `PZ234918.1`, `PZ234914.1`, `PZ234913.1`, `PZ234912.1`, `PZ234911.1`, `PZ234910.1`, `PZ234909.1` |
| Dengue-2 | 10 | `NC_001474.2`, `PZ603444.1`, `PZ603442.1`, `PZ234917.1`, `PZ234916.1`, `PZ234915.1`, `PZ234902.1`, `PZ234901.1`, `PZ234900.1`, `PZ234899.1` |
| Dengue-3 | 10 | `NC_001475.2`, `LC773256.1`, `LC773253.1`, `PZ546417.1`, `PZ516052.1`, `OZ484517.1`, `OZ484516.1`, `OZ484515.1`, `OZ484513.1`, `OZ484500.1` |
| Dengue-4 | 10 | `NC_002640.1`, `OZ484504.1`, `OZ484503.1`, `PZ049732.1`, `PX480447.1`, `PX480446.1`, `PX480445.1`, `PX480444.1`, `PX480443.1`, `PX480442.1` |
| Zika | 10 | `NC_012532.1`, `NC_035889.1`, `PV012995.1`, `PV012994.1`, `PX404510.1`, `PX404503.1`, `PX404502.1`, `PX404501.1`, `PX404500.1`, `PX404475.1` |
| West-Nile | 10 | `NC_009942.1`, `NC_001563.2`, `PZ437308.1`, `PZ437307.1`, `PZ437306.1`, `PZ437305.1`, `PZ437304.1`, `PX404496.1`, `PX404495.1`, `PV693361.1` |
| Yellow-Fever | 10 | `NC_002031.1`, `PX766933.1`, `PX766931.1`, `PX766930.1`, `PX766929.1`, `PX766928.1`, `PX766926.1`, `PX766925.1`, `PX766924.1`, `PX766923.1` |
| Japanese-Encephalitis | 10 | `NC_001437.1`, `PZ123354.1`, `PZ345904.1`, `PX400461.1`, `PV804250.1`, `PV804249.1`, `PV666603.1`, `PV666602.1`, `PV666601.1`, `PV666600.1` |

### Filoviridae

8 classes, 64 isolates.

| Class | Isolates | Accessions |
|---|---|---|
| Ebola-Zaire | 10 | `NC_002549.1`, `OZ276552.1`, `OR084932.1`, `OR084931.1`, `OR084930.1`, `OR084929.1`, `OR084928.1`, `OR084927.1`, `OR084926.1`, `OR084925.1` |
| Ebola-Sudan | 10 | `NC_006432.1`, `OZ222062.1`, `OQ673069.1`, `OQ673068.1`, `OQ673067.1`, `OQ673066.1`, `OQ673065.1`, `OQ673064.1`, `OQ673063.1`, `OQ673062.1` |
| Ebola-Bundibugyo | 10 | `NC_014373.1`, `OZ492176.1`, `OZ491541.1`, `OZ491540.1`, `OZ491539.1`, `OZ491538.1`, `OZ491537.1`, `OZ491536.1`, `OZ491535.1`, `OZ491534.1` |
| Ebola-TaiForest | 3 | `NC_014372.1`, `MH121167.1`, `KU182910.1` |
| Ebola-Reston | 10 | `NC_004161.1`, `MT796851.1`, `MK088530.1`, `MF540571.1`, `MF540570.1`, `KY008770.1`, `KY798012.1`, `KY798011.1`, `KY798010.1`, `KY798009.1` |
| Marburg | 9 | `NC_001608.4`, `PX278686.1`, `PX278685.1`, `KP985768.1`, `KY047764.1`, `KY047763.1`, `KR867677.1`, `KP117261.1`, `KP117260.1` |
| Ravn | 6 | `NC_024781.1`, `KU179482.1`, `EU500828.1`, `EU500827.1`, `EU500826.1`, `EF446131.1` |
| Lloviu | 6 | `NC_016144.1`, `OQ630505.1`, `ON186772.1`, `MZ541881.1`, `MW775011.1`, `MW775010.1` |

### Hepadnaviridae

8 classes, 71 isolates.

| Class | Isolates | Accessions |
|---|---|---|
| HBV-genotype-A | 10 | `LC784044.1`, `LC778006.1`, `LC778005.1`, `LC778004.1`, `LC773628.1`, `LC773627.1`, `LC773620.1`, `LC533934.1`, `LC519823.1`, `LC519820.1` |
| HBV-genotype-B | 10 | `LC784057.1`, `LC784049.1`, `LC784048.1`, `LC784047.1`, `LC784046.1`, `LC784045.1`, `LC778015.1`, `LC778014.1`, `LC778013.1`, `LC778012.1` |
| HBV-genotype-C | 10 | `LC886239.1`, `LC886238.1`, `LC784056.1`, `LC784055.1`, `LC784054.1`, `LC784053.1`, `LC784052.1`, `LC784051.1`, `LC784050.1`, `LC778017.1` |
| HBV-genotype-D | 10 | `LC852229.1`, `LC519824.1`, `LC519816.1`, `LC519814.1`, `LC519810.1`, `LC519809.1`, `LC519808.1`, `LC519802.1`, `LC519795.1`, `LC519793.1` |
| HBV-genotype-E | 10 | `PP445059.1`, `PP445056.1`, `PP445055.1`, `PP445053.1`, `LC773632.1`, `LC513657.1`, `LC513656.1`, `LC513654.1`, `LC513652.1`, `AP007262.1` |
| HBV-genotype-F | 3 | `HE974369.1`, `HE974368.1`, `HE974366.1` |
| HBV-genotype-G | 8 | `AB625343.1`, `AB625342.1`, `AP007264.1`, `HE981176.1`, `HE981174.1`, `HE981173.1`, `HE981172.1`, `HE981171.1` |
| HBV-genotype-H | 10 | `LC491577.1`, `AB846650.1`, `AB516395.1`, `AB516394.1`, `AB516393.1`, `AB298362.1`, `AP007261.1`, `AY090454.1`, `AB179747.1`, `AB275308.1` |

### Retroviridae

3 classes, 21 isolates.

| Class | Isolates | Accessions |
|---|---|---|
| HIV-1-subtype-B | 7 | `AF075719.1`, `M17451.1`, `AF004394.1`, `U21135.1`, `K02013.1`, `K03455.1`, `M17449.1` |
| HIV-1-subtype-C | 4 | `AB023804.1`, `AY162225.1`, `AY162224.1`, `AY162223.1` |
| HIV-2 | 10 | `NC_001722.1`, `LC927254.1`, `LC927253.1`, `AB100245.1`, `M30895.1`, `AB731745.1`, `AB731743.1`, `AB731742.1`, `AB731741.1`, `AB731740.1` |

### Orthoherpesviridae (retrieved as *Herpesviridae*)

8 classes, 76 isolates.

| Class | Isolates | Accessions |
|---|---|---|
| HSV-1 | 10 | `NC_001806.2`, `PZ169719.1`, `PZ169717.1`, `PZ169716.1`, `PZ169713.1`, `PZ169711.1`, `PZ169710.1`, `PZ169709.1`, `PZ169698.1`, `PZ169696.1` |
| HSV-2 | 10 | `NC_001798.2`, `PZ135825.1`, `OM370995.1`, `MK855052.1`, `MN187895.1`, `LT799380.1`, `LT797786.1`, `LT797682.1`, `LT797636.1`, `LT797635.1` |
| VZV | 10 | `NC_001348.1`, `PV061461.1`, `PQ641243.1`, `PQ505474.1`, `PP261332.1`, `PP261331.1`, `PP386307.1`, `PP169944.1`, `OQ916050.1`, `OQ916049.1` |
| EBV | 10 | `NC_009334.1`, `NC_007605.1`, `OR134772.1`, `OR134771.1`, `OR134770.1`, `OR134769.1`, `OR134768.1`, `OR134767.1`, `OR134766.1`, `OR134765.1` |
| HCMV | 10 | `NC_006273.2`, `PZ137009.1`, `LC846339.1`, `LC846338.1`, `PX246278.1`, `PQ851844.1`, `PQ851843.1`, `PQ851842.1`, `PQ851841.1`, `PQ851840.1` |
| HHV-6A | 10 | `NC_001664.4`, `PP921919.1`, `MW049322.1`, `MW049321.1`, `MW049319.1`, `MW049318.1`, `MW049317.1`, `MW049316.1`, `MW049315.1`, `MW049314.1` |
| HHV-6B | 6 | `NC_000898.1`, `OR643889.1`, `MW536483.1`, `MH698402.1`, `MH698401.1`, `AB021506.1` |
| HHV-7 | 10 | `NC_001716.2`, `PV694273.1`, `PV694272.1`, `PV694271.1`, `PQ541394.1`, `OR634980.1`, `OR634979.1`, `ON023041.1`, `ON023040.1`, `ON023039.1` |

### Picornaviridae

8 classes, 80 isolates.

| Class | Isolates | Accessions |
|---|---|---|
| Poliovirus-1 | 10 | `PP506141.1`, `PP506140.1`, `PP506139.1`, `OR129532.1`, `PP068134.1`, `PP068131.1`, `OR253701.1`, `OQ286212.1`, `OQ286211.1`, `OQ286208.1` |
| Poliovirus-2 | 10 | `PX725683.1`, `PX725682.1`, `PX725681.1`, `PX725680.1`, `PX725679.1`, `PX725678.1`, `PX725677.1`, `PX725676.1`, `PX725675.1`, `PX725674.1` |
| Poliovirus-3 | 10 | `PZ134350.1`, `PP356987.1`, `PP356986.1`, `PP348560.1`, `PP348559.1`, `PP348558.1`, `PP533225.1`, `PP533224.1`, `PP533223.1`, `PP533222.1` |
| Coxsackievirus-A16 | 10 | `PV173118.1`, `PV173117.1`, `PV173116.1`, `PV173115.1`, `PV173114.1`, `PV173113.1`, `PV173112.1`, `OQ091687.1`, `OQ091686.1`, `OQ091685.1` |
| Coxsackievirus-B3 | 10 | `NC_038307.1`, `PZ117059.1`, `PZ117057.1`, `PV863042.1`, `PV863041.1`, `PV863040.1`, `PV863039.1`, `PV863038.1`, `PV863037.1`, `PV863036.1` |
| Rhinovirus-A | 10 | `NC_038311.1`, `NC_001617.1`, `PZ233729.1`, `PZ233728.1`, `PZ233726.1`, `PZ233721.1`, `PZ233720.1`, `PZ233719.1`, `PZ233717.1`, `PZ233716.1` |
| Rhinovirus-B | 10 | `NC_001490.1`, `NC_038312.1`, `PZ233731.1`, `PZ233730.1`, `PX404530.1`, `PX404528.1`, `PX404526.1`, `PX404525.1`, `PX404521.1`, `PX404518.1` |
| Rhinovirus-C | 10 | `NC_038878.1`, `NC_009996.1`, `PZ496045.1`, `PZ496041.1`, `PZ233732.1`, `PZ233727.1`, `PZ233725.1`, `PZ233724.1`, `PZ233723.1`, `PZ233722.1` |

### Paramyxoviridae

8 classes, 80 isolates.

| Class | Isolates | Accessions |
|---|---|---|
| Measles | 10 | `NC_001498.1`, `PZ562648.1`, `PZ569392.1`, `PZ569391.1`, `PZ569390.1`, `PZ569389.1`, `PZ569388.1`, `PZ569387.1`, `PZ569386.1`, `PZ569385.1` |
| Mumps | 10 | `NC_002200.1`, `PX475736.1`, `PV668547.1`, `PV668546.1`, `PV668541.1`, `PV668540.1`, `PV668539.1`, `PV668538.1`, `PV668537.1`, `PV668536.1` |
| RSV-A | 10 | `NC_001803.1`, `PZ612962.1`, `PZ612961.1`, `PZ612960.1`, `PZ612959.1`, `PZ612958.1`, `PZ612957.1`, `PZ612956.1`, `PZ612955.1`, `PZ612954.1` |
| RSV-B | 10 | `PZ575938.1`, `PZ575937.1`, `PZ575936.1`, `PZ575935.1`, `PZ575934.1`, `PZ575933.1`, `PZ575932.1`, `PZ575931.1`, `PZ575930.1`, `PZ575929.1` |
| Human-Parainfluenza-1 | 10 | `NC_003461.1`, `LC817390.1`, `LC817389.1`, `LC720862.1`, `PX380303.1`, `PX380302.1`, `PX380301.1`, `PX380300.1`, `PX380299.1`, `PX380298.1` |
| Human-Parainfluenza-3 | 10 | `NC_075446.1`, `NC_001796.2`, `LC817398.1`, `LC720880.1`, `PZ244842.1`, `PZ244839.1`, `PZ244838.1`, `PZ244837.1`, `LC922678.1`, `LC922672.1` |
| Nipah | 10 | `NC_002728.1`, `PV640869.1`, `PX130166.1`, `PX130165.1`, `PX130164.1`, `PV892943.1`, `PV132707.1`, `PV132706.1`, `PQ463988.1`, `PQ368169.1` |
| Human-Metapneumovirus | 10 | `NC_039199.1`, `PZ239602.1`, `PZ239601.1`, `PZ399301.1`, `PZ399300.1`, `PZ399299.1`, `PZ399298.1`, `PZ399297.1`, `PZ399296.1`, `PZ399295.1` |

## Naming

Family names are those used when the genomes were retrieved, so that the stored paths and manifests stay valid. Where ICTV now uses a different name, the current one is given above and the retrieval-time name is noted beside it.

