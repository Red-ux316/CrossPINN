# Encoder-Decoder TransPINN (Operator Learning)

Questa architettura rappresenta l'approccio più rigoroso e potente del nostro MLOps pipeline, in linea con lo stato dell'arte dell'Operator Learning Fisico (es. *Perceiver IO* o *GNOT*). È progettata per gestire punti di collocazione sparsi (es. generati dinamicamente tramite Latin Hypercube Sampling - LHS) e risolvere il problema della non-località mantenendo i costi di memoria invarianti rispetto alla densità di integrazione.

## Architettura e Flusso Dati
L'architettura è divisa in due blocchi funzionali distinti, che implementano la **Codifica Spaziale Simmetrica** per garantire una metrica di distanza geometricamente significativa.

### 1. L'Encoder Globale (Self-Attention)
A differenza del TLP, il TransPINN genera la sua memoria latente dinamicamente a partire da una griglia di punti di supporto.
*   **Input**: Un set di $N$ punti di supporto (*Support Points*) sparsi nel dominio normalizzato `[0, 1]`.
*   **Codifica Posizionale**: I punti di supporto vengono codificati da un `pos_encoder` configurabile (es. `LFF`, `RFF`, `MLP`, `SIREN`).
*   **Self-Attention**: I vettori latenti risultanti vengono processati da layer multipli di **Self-Attention** (in modalità **Pre-LayerNormalization**). Tutti i punti di supporto comunicano tra loro, costruendo una mappa latente globale che cattura la struttura del dominio.
*   **Output**: La mappa latente globale, che funge da **Key (K)** e **Value (V)** per il decoder.

### 2. Il Decoder Continuo (Cross-Attention)
Il `PINNDecoder` è un modulo condiviso con il TLP e ha il compito di interrogare la mappa latente.
*   **Input**: Riceve le coordinate continue `x_query` (i punti di collocazione) e la mappa latente `(K, V)` dall'encoder.
*   **Codifica Simmetrica**: Il decoder utilizza lo **stesso identico `pos_encoder`** dell'encoder per mappare le coordinate `x_query` nello stesso spazio latente, generando le **Query (Q)**.
*   **Cross-Attention**: Esegue la Cross-Attention tra le Query (Q) e la mappa latente (K, V) per estrarre l'informazione contestuale.
*   **Maschera Causale Esterna (Opzionale)**: Se abilitata (`causal_mask: true`), una funzione fisica (`mask_fn`) oscura le interazioni non permesse, forzando il rispetto dei limiti di integrazione di Volterra.
*   **Output**: Genera le previsioni $\hat{u}$ esatte per tutti i punti continui richiesti.

### 3. Opzioni per il Positional Encoder
Il TransPINN condivide la stessa factory di encoder posizionali del TLP, permettendo di sperimentare diverse strategie di codifica spaziale:
1.  **Random Fourier Features (RFF)**: Frequenze fisse, garantisce una metrica euclidea.
2.  **Learnable Fourier Features (LFF)**: Frequenze apprendibili, unisce geometria e flessibilità.
3.  **Mini-MLP Classico**: Nessun *inductive bias*, ma soffre di *Spectral Bias*.
4.  **MLP Sinusoidale (SIREN)**: Supera lo *Spectral Bias* grazie all'attivazione sinusoidale.
5.  **Radial Basis Functions (RBF)**: Ideale per modellare discontinuità locali.

## Perché supera RISN e Causal History
Questa architettura risolve il problema della **fissità dimensionale**. Il Decoder a Cross-Attention agisce come un interpolatore neurale globale: permette di interrogare la rete in un numero arbitrario di punti di Gauss ($M$) senza mai far esplodere la memoria in $\mathcal{O}(M^2)$. La complessità dell'Attention è limitata a $\mathcal{O}(M \times N)$, dove $N$ (i punti di supporto) è un iperparametro fisso e limitato.

Inoltre, mantiene intatto l'*inductive bias* **non-locale**: ogni predizione, anche nel punto di Gauss più remoto o asimmetrico, è calcolata "guardando" alla mappa globale dell'intero dominio, rendendolo l'architettura ideale sia per le Equazioni di Volterra (con limiti dinamici) che per quelle di Fredholm (con limiti globali).

## Vantaggi e Svantaggi
* **Pro:** Risolve del tutto la *Gradient Stiffness* tipica delle MLP permettendo al contempo una visione globale perfetta che le Sequenze Causali non hanno. È *resolution-invariant* in fase di inferenza: tratta le coordinate di test come query puramente spaziali indipendenti da indici o *mesh* prefissate.
* **Contro:** L'Encoder-Decoder necessita obbligatoriamente di algoritmi di ottimizzazione compatibili con gradienti non convessi (es. **AdamW**) accompagnati da *Learning Rate Schedulers* (es. Cosine Annealing), poiché l'Encoder e il Decoder devono imparare ad allineare il proprio spazio latente ("parlarsi") prima di poter minimizzare efficacemente la fisica. Fallisce matematicamente con ottimizzatori classici puramente convessi come L-BFGS.
