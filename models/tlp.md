# Architettura TLP: Memoria Simmetrica e Addestramento a Due Fasi

## 1. Introduzione e Motivazione Teorica
Il modello **TLP (Trans Lite PINN)** è un'evoluzione diretta dell'architettura TransPINN standard, ottimizzata per la risoluzione di Equazioni Integro-Differenziali (come le equazioni di Volterra). 

L'architettura nasce dall'osservazione che, nel contesto del training di una PINN su un singolo dominio, l'Encoder di un Transformer classico agisce come un organo computazionalmente ridondante. Ricevendo in input una griglia statica di punti di supporto, l'Encoder impiega calcoli di Self-Attention onerosi per generare un *Positional Encoding* complesso ma fondamentalmente statico.

TLP risolve questo collo di bottiglia eliminando completamente il blocco Encoder e sostituendolo con una **Memoria Latente Apprendibile** (Learnable Latent Memory), ispirandosi alle architetture *Perceiver IO* e *DeepONet*.

---

## 2. Architettura Unificata

### 2.1 Codifica Spaziale Simmetrica (Il "Dot-Product Distance Trap")
Il cuore del TLP è la **Codifica Spaziale Simmetrica**. In un Transformer, la "distanza" tra punti è misurata dal prodotto scalare `Q·Kᵀ`. Se le Query (punti di collocazione) e le Key (punti di supporto della memoria) sono codificate in modi diversi, questa metrica di distanza collassa.

La soluzione è un `pos_encoder` condiviso che mappa sia le coordinate delle Query che quelle della Memoria nello stesso spazio latente, garantendo che l'Attention misuri una distanza geometrica significativa e simmetrica.

### 2.2 Opzioni per il Positional Encoder
Il `pos_encoder` è configurabile per adattarsi a diverse fisiche. Le 5 opzioni principali sono:

1.  **Random Fourier Features (RFF) - *Statico***: Usa frequenze fisse campionate da una Gaussiana. Garantisce matematicamente che l'Attention misuri la distanza euclidea, ma non si adatta alla PDE.

2.  **Learnable Fourier Features (LFF) - *Apprendibile & Trigonometrico***: Le frequenze sono parametri `nn.Parameter`. Unisce la geometria trigonometrica con la flessibilità di apprendere le scale spaziali rilevanti.

3.  **Mini-MLP Classico - *Apprendibile & Non-Lineare***: Un piccolo MLP a due strati con attivazione GeLU. Nessun *inductive bias* geometrico, ma soffre di *Spectral Bias* (fatica a imparare alte frequenze).

4.  **MLP Sinusoidale (SIREN) - *Apprendibile & Periodico***: Sostituisce la GeLU con un'attivazione `sin(ωx)`, distruggendo lo *Spectral Bias* e permettendo di codificare dettagli ad altissima frequenza.

5.  **Radial Basis Functions (RBF) - *Apprendibile & Localizzato***: Usa centri gaussiani apprendibili. Eccellente per modellare discontinuità o shock spaziali.

### 2.3 Architettura della Memoria Simmetrica
A differenza di un'architettura a due percorsi (uno per il cold-start e uno per il warm-start), il TLP adotta un'architettura **unificata**. La memoria latente è *sempre* costruita simmetricamente come somma di due componenti: una posizionale e una di valore.

1.  **Componente Posizionale (Il "Dove")**: Una griglia fissa di punti di supporto $x_{supp}$ (generata nel dominio normalizzato `[0, 1]`) viene codificata dal `pos_encoder` condiviso:
    $$ \text{mem\_pos} = \text{pos\_encoder}(x_{supp}) $$

2.  **Componente di Valore (Il "Cosa")**: Un parametro apprendibile `u_learnable` (un tensore di dimensioni `[N_support, 1]`) rappresenta i valori della soluzione sui punti di supporto. Viene codificato da un `val_encoder` dedicato:
    $$ \text{mem\_val} = \text{val\_encoder}(\text{u\_learnable}) $$

3.  **Fusione**: Le due componenti vengono sommate per creare la memoria finale, che funge da Key (K) e Value (V) per la Cross-Attention:
    $$\text{Memoria} = \text{mem\_pos} + \text{mem\_val}$$

### 2.4 Strategia di Addestramento: Cold-Start vs Warm-Start a 2 Fasi
La flessibilità del TLP risiede in come viene gestito il parametro `u_learnable`.

*   **Cold-Start (Addestramento da Zero)**: Se `training.warm_start: false`, il parametro `u_learnable` viene inizializzato casualmente e i suoi gradienti sono attivi (`requires_grad=True`) fin dall'epoca 1. Il modello impara l'intera soluzione da zero.

*   **Warm-Start (Addestramento a Due Fasi)**: Se `training.warm_start: true` e un modello MLP pre-addestrato è disponibile, l'addestramento si divide in due fasi:
    1.  **Fase 1 (Prior Congelato)**: `u_learnable` viene inizializzato con i valori della soluzione MLP e i suoi gradienti vengono "congelati" (`requires_grad=False`). In questa fase, il modello impara solo gli encoder (`pos_encoder`, `val_encoder`) e il decoder, usando il prior della MLP come base fissa.
    2.  **Fase 2 (Fine-Tuning)**: All'epoca definita da `warm_start_unfreeze_epoch`, il `Trainer` "scongela" `u_learnable` (`requires_grad=True`). Ora il modello può affinare con precisione chirurgica i valori della soluzione sui punti di supporto per minimizzare ulteriormente il residuo della PDE.

### 2.5 Decoder e Gestione delle Query
Il TLP delega la gestione delle query al `PINNDecoder`. Quando il modello riceve un batch di coordinate continue $x_{query}$ (i punti di collocazione), le passa direttamente al decoder.

Il `PINNDecoder` si occupa internamente di:
1.  Normalizzare le coordinate `x_query`.
2.  Codificarle usando lo **stesso identico `pos_encoder`** della memoria per generare le **Query (Q)**.
3.  Eseguire la Cross-Attention tra le Query (Q) e la Memoria (K, V).

Grazie all'encoder condiviso, l'operazione di prodotto scalare $Q \cdot K^T$ valuta una distanza geometricamente significativa, permettendo alla Softmax di produrre picchi di attenzione solo sui punti di supporto adiacenti alla coordinata interrogata.

---

## 3. Vantaggi Fisici e Computazionali

Rispetto a un TransPINN completo, il TLP offre vantaggi cruciali:

1. **Abbattimento della Complessità Computazionale:**
   L'Encoder standard richiede una Self-Attention di ordine $O(N_{support}^2)$. Sostituendolo con una memoria latente, il costo della codifica crolla a $O(1)$ durante il *forward pass*. L'onere computazionale rimane solo nella Cross-Attention del Decoder, che è $O(M \times N_{support})$.
2. **Efficienza in VRAM e Risoluzione Dati:**
   L'eliminazione dei layer dell'Encoder libera VRAM, che può essere reinvestita aumentando il numero di punti di collocazione (migliorando i gradienti stocastici) o i nodi di quadratura.
3. **Semplificazione del Loss Landscape:**
   Nel TLP, lo spazio di ricerca dei pesi è più snello e direttamente accoppiato al residuo della PDE, riducendo la probabilità che il modello si stabilizzi in minimi locali non ottimali.
