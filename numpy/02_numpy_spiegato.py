# %% [markdown]
# # NumPy: ripasso della lezione 2 (ML4N, Politecnico di Torino)
#
# Segue l'ordine della lezione `02_numpy.py` del prof. Galante.
#
# **Come usarlo in VS Code:** ogni blocco che inizia con `# %%` è una *cella*.
# Cliccaci dentro e premi **Shift+Enter**: VS Code la esegue nella
# "Interactive Window" e ti mostra l'output (serve l'estensione Python + Jupyter
# e `pip install numpy ipykernel` nel tuo venv).
#
# **Metodo:** dove trovi `# PREVEDI:` fermati, scrivi su carta cosa ti aspetti
# (valore, shape, errore sì/no), POI esegui. Se sbagli la previsione, è
# esattamente lì che devi ripassare. Leggere e basta non serve a nulla.

# %% [markdown]
# ---
# ## 1. NumPy ed efficienza
#
# Un **array** NumPy è:
# - **a tipo fisso**: tutti gli elementi hanno lo stesso `dtype`;
# - **contiguo in memoria**: i valori stanno uno accanto all'altro (8 byte per un `float64`).
#
# Una **lista** Python invece contiene *riferimenti* a oggetti sparsi in memoria,
# ognuno con il suo header: un `float` in lista costa ~32 byte (8 di riferimento
# + 24 dell'oggetto). E a ogni operazione Python deve controllare il tipo di
# ogni singolo elemento.
#
# Risultato: NumPy usa ~4 volte meno memoria ed è molto più veloce, perché il
# ciclo gira in codice compilato ("vettorizzato"), non in Python.

# %%
import sys
import time

import numpy as np  # convenzione universale: chiamalo sempre np

print("versione NumPy:", np.__version__)

arr = np.array([0.67, 0.45, 0.33])
print(arr, arr.dtype, arr.shape)

# %%
# Prova tu la differenza di velocità: prodotto scalare di due vettori da 1 milione
n = 1_000_000
a_list = [float(i) for i in range(n)]
b_list = [float(i) for i in range(n)]
a_arr = np.array(a_list)
b_arr = np.array(b_list)

t0 = time.perf_counter()
dot_list = sum(x * y for x, y in zip(a_list, b_list))
t1 = time.perf_counter()
dot_arr = np.dot(a_arr, b_arr)
t2 = time.perf_counter()

print(f"lista: {t1 - t0:.4f} s   numpy: {t2 - t1:.4f} s   (x{(t1 - t0) / (t2 - t1):.0f} più veloce)")
print("uguali con ==?", dot_list == dot_arr)  # probabilmente False!
print("uguali con isclose?", np.isclose(dot_list, dot_arr))
# I float sommati in ordine diverso danno arrotondamenti diversi: tra float non
# si usa mai ==, si usa np.isclose / np.allclose.

# %%
# E la memoria (10 000 float)
lst = [float(i) for i in range(10_000)]
mem_list = sys.getsizeof(lst) + sum(sys.getsizeof(v) for v in lst)
mem_arr = np.arange(10_000, dtype=np.float64).nbytes
print(f"lista: {mem_list} byte   array: {mem_arr} byte")

# %% [markdown]
# ---
# ## 2. Array: dimensioni, shape, dtype, creazione
#
# ### Assi e shape
# - Gli **assi** sono le dimensioni: vettore 1-D = 1 asse, matrice = 2 assi, ecc.
# - La **shape** è la tupla con il numero di elementi lungo ogni asse.
# - Ogni nuova dimensione si aggiunge **a sinistra** della shape: diventa l'asse 0.
#   Quindi per una matrice la shape è `(righe, colonne)`, per un 3D è `(profondità, righe, colonne)`.
# - **L'asse -1 è sempre l'ultimo**: si muove *lungo una riga* (attraverso le colonne), qualunque sia il numero di dimensioni.

# %%
arr1 = np.array([1, 2, 3])
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6]])
arr3 = np.array([[[1, 2, 3], [4, 5, 6]],
                 [[7, 8, 9], [10, 11, 12]],
                 [[13, 14, 15], [16, 17, 18]]])

# PREVEDI: le tre shape, prima di eseguire
print(arr1.shape)
print(arr2.shape)
print(arr3.shape)

# %% [markdown]
# `(3,)` e non `(3)`: la virgola è ciò che rende una tupla una tupla (lezione 1).
#
# ### dtype
# - Interi: `int8 ... int64` (il numero è la larghezza **in bit**), `uint8 ...` senza segno
# - Float: `float16`, `float32`, `float64` (default per i decimali)
# - `bool`: 1 byte per elemento, è ciò di cui sono fatte le maschere
#
# Il dtype viene **dedotto** dai valori; se sono misti NumPy sceglie un tipo che
# li contenga tutti (**upcasting**).

# %%
# PREVEDI: il dtype di ciascuno
print(np.array([0.0, 1.0, 2.0]).dtype)
print(np.array([0, 1, 2]).dtype)
print(np.array([1, 2.5]).dtype)
print(np.array([0.0, 1.0, 2.0], dtype=np.float16).dtype)
print(np.array([True, False]).dtype)
print(np.array([1, True]).dtype)  # domanda trabocchetto: bool + int?

# %% [markdown]
# ### Vettore 1-D vs vettore colonna
# Questa distinzione è **la** fonte degli errori di broadcasting più avanti.
# - `np.array([0.1, 0.2, 0.3])` ha shape `(3,)`: un solo asse, si comporta "come una riga".
# - `np.array([[0.1], [0.2], [0.3]])` ha shape `(3, 1)`: è una **matrice 2D** con una colonna.
# - Una riga "vera" 2D ha shape `(1, 3)`.

# %%
b = np.array([0.1, 0.2, 0.3])
a = np.array([[0.1], [0.2], [0.3]])
r = np.array([[0.1, 0.2, 0.3]])
print(b.shape, b.ndim)
print(a.shape, a.ndim)
print(r.shape, r.ndim)

# %% [markdown]
# ### Creare array da zero
# | Funzione | Cosa ottieni |
# |---|---|
# | `np.zeros(shape)`, `np.ones(shape)` | tutti 0 / tutti 1 (float64!) |
# | `np.full(shape, value)` | tutti uguali a `value` |
# | `np.linspace(start, stop, num)` | `num` campioni, **stop incluso** |
# | `np.arange(start, stop, step)` | come `range`, **stop escluso** |
# | `np.random.random(shape)` | uniformi in [0, 1) |
# | `np.random.normal(mean, std, shape)` | gaussiani |
#
# Nota: la shape si passa come **tupla**: `np.ones((2, 3))`, non `np.ones(2, 3)`.

# %%
print(np.ones((2, 3)))
print(np.full((2, 1), 1.1))
print(np.linspace(0, 1, 11))  # PREVEDI: quanti elementi? l'1 c'è?
print(np.arange(1, 7, 2))     # PREVEDI: il 7 c'è?

np.random.seed(2)  # senza seed i numeri cambiano a ogni esecuzione
print(np.random.random((2, 3)))
print(np.random.normal(0, 1, (2, 3)))

# %%
# Attributi principali
x = np.array([[2, 3, 4], [5, 6, 7]])
print("ndim:", x.ndim)    # numero di dimensioni
print("shape:", x.shape)  # come sono disposti
print("size:", x.size)    # quanti numeri in totale = prodotto della shape
print(np.ndim(x), np.shape(x), np.size(x))  # le stesse, come funzioni

# %% [markdown]
# **Esercizi del prof da fare ora:** `2.1_numpy_arrays.ipynb`, esercizi 1, 2, 3.

# %% [markdown]
# ---
# ## 3. Calcolo con gli array
#
# Quattro tipi di operazioni, tutte **senza cicli Python**:
# | Tipo | Esempi | Elemento per elemento? | Output |
# |---|---|---|---|
# | ufunc binaria | `x + y`, `x * y`, `x ** 2`, `x // y`, `x % y` | sì | stessa shape |
# | ufunc unaria | `np.exp(x)`, `np.log(x)`, `np.abs(x)`, `np.sin(x)` | sì | stessa shape, `x` **non** modificato |
# | aggregazione | `x.sum()`, `x.mean()`, `x.std()`, `x.argmax()` | no | un valore, o un asse in meno con `axis=` |
# | algebra | `np.dot(x, y)`, `x @ y` | no | shape del prodotto matriciale |

# %%
x = np.array([[1, 1], [2, 2]])
y = np.array([[3, 4], [6, 5]])
print(x * y)   # PREVEDI: è il prodotto matriciale? (NO!)
print(x + y)
print(x ** 2)
print(x @ y)   # questo sì è il prodotto matriciale: confronta con x * y

# %%
print(np.exp(x))
print(np.log2(np.array([1.0, 2.0, 4.0, 8.0])))
print(x)  # invariato: una ufunc restituisce sempre un array NUOVO

# %% [markdown]
# ### Aggregazioni
# `np.sum(x)` e `x.sum()` sono equivalenti (lo stesso per min, max, mean, std, argmin, argmax).
#
# - `x.std()` divide per **n** (deviazione standard di popolazione). Per quella
#   campionaria serve `x.std(ddof=1)`. Attenzione: pandas usa ddof=1 di default,
#   NumPy ddof=0. È una differenza che fa sbagliare nei lab.
# - `argmax()` senza asse restituisce la **posizione** nell'array letto riga per
#   riga ("appiattito"), non il valore.

# %%
x = np.array([[1, 1], [2, 2]])
print(x.sum(), x.mean(), x.std(), x.std(ddof=1))
print(x.argmax())  # PREVEDI: 2, cioè la posizione piatta del primo 2
print(np.unravel_index(x.argmax(), x.shape))  # extra: riconverte in (riga, colonna)

# %% [markdown]
# ### Aggregare lungo un asse: LA regola
# `axis=k` dice lungo quale dimensione si aggrega, e **quella dimensione sparisce dall'output**.
#
# - Matrice `(2, 3)` con `axis=0` → collassa le righe → shape `(3,)`: un valore **per colonna**.
# - Matrice `(2, 3)` con `axis=1` (o `-1`) → collassa le colonne → shape `(2,)`: un valore **per riga**.
#
# Trucco mnemonico: non pensare "sommo le righe", pensa "**quale asse elimino dalla shape**".
# È anche il modo per verificare: guarda la shape di output.

# %%
x = np.array([[1, 7], [2, 4]])
print(x.sum(axis=-1), x.sum(axis=-1).shape)  # PREVEDI: una somma per riga
print(x.sum(axis=0))                          # PREVEDI: una somma per colonna
print(x.argmax(axis=0))                       # indice della riga col massimo, per ogni colonna

# 3D: la regola non cambia
c = np.arange(24).reshape(2, 3, 4)
print(c.shape)
print(c.sum(axis=0).shape)   # PREVEDI
print(c.sum(axis=1).shape)   # PREVEDI
print(c.sum(axis=-1).shape)  # PREVEDI

# %% [markdown]
# ### Ordinamento
# - `np.sort(x)` → **copia** ordinata, `x` non cambia.
# - `x.sort()` → ordina **in place** e restituisce `None` (errore classico: `y = x.sort()` dà `y = None`).
# - Default lungo l'**asse -1**: ogni riga ordinata per conto suo. Ordinare una matrice non ordina "tutto".
# - `np.argsort(x)` → le **posizioni** che ordinerebbero l'array. Serve per ordinare
#   *qualcos'altro* in base a questi valori (es. etichette in base ai punteggi).

# %%
x = np.array([[2, 1, 3], [7, 9, 8]])
print(np.sort(x))
print(x)  # invariato
other = np.array([[2, 7, 3], [7, 2, 1]])
print(np.sort(other, axis=0))  # ogni colonna ordinata per conto suo

v = np.array([30, 10, 20])
order = np.argsort(v)
print(order)     # PREVEDI
print(v[order])  # ordinato
labels = np.array(["c", "a", "b"])
print(labels[order])  # etichette ordinate secondo v: il pattern tipico

# %% [markdown]
# ### Algebra
# `np.dot` fa: prodotto scalare fra vettori 1-D, matrice × vettore, matrice × matrice.
# Da Python 3.5 si può usare `@`, più leggibile nelle formule.

# %%
x = np.array([1, 2, 3])
y = np.array([0, 2, 1])
print(np.dot(x, y))  # 1*0 + 2*2 + 3*1
m = np.array([[1, 1], [2, 2]])
print(np.dot(m, np.array([2, 3])))
print(np.dot(m, np.array([[2, 2], [1, 1]])))
print(m @ np.array([[2, 2], [1, 1]]))

# %% [markdown]
# **Esercizi del prof da fare ora:** `2.2_numpy_operations.ipynb`, esercizi 1, 2, 3.

# %% [markdown]
# ---
# ## 4. Broadcasting
#
# Permette operazioni fra array di shape diverse **senza copiare dati**.
#
# **Le 3 regole** (le shape si confrontano **da destra**):
# 1. Se un array ha meno dimensioni, la sua shape viene completata con **1 a sinistra**.
# 2. Se lungo un asse uno ha dimensione **1** e l'altro **> 1**, quello da 1 viene **stirato**.
# 3. Se lungo un asse le dimensioni sono **diverse ed entrambe > 1** → **errore**.
#
# Procedura da fare a mano (fatela sempre, all'esame e nei lab):
# ```
# x: (3,)     -> regola 1 -> (1, 3)
# y: (3, 1)                  (3, 1)
#                            ------
#                  regola 2  (3, 3)   ok
# ```

# %%
x = np.array([1, 2, 3])
print(x + 10)  # caso più semplice: lo scalare viene esteso a ogni cella

y = np.array([[11], [12], [13]])
print(x.shape, y.shape)
z = x + y
print(z, z.shape)  # PREVEDI la shape prima di eseguire

# %%
# Quando fallisce
x = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
y = np.array([11, 12, 13])              # (3,) -> (1, 3)
try:
    x + y
except ValueError as e:
    print("ERRORE:", e)
# Confronto da destra: 2 vs 3 -> diversi ed entrambi > 1 -> regola 3.

# %%
# Come si ripara? Rendendo y una colonna (3, 1): allora (3,2) e (3,1) -> (3,2)
print(x + y.reshape(-1, 1))
print(x + y[:, np.newaxis])  # stessa cosa, sintassi alternativa (np.newaxis = None)

# %%
# PREVEDI per ciascuna: shape del risultato oppure errore
casi = [((4, 3), (3,)), ((4, 3), (4,)), ((4, 1), (1, 5)), ((2, 1, 3), (4, 1)), ((5,), (5, 1))]
for s1, s2 in casi:
    try:
        out = (np.zeros(s1) + np.zeros(s2)).shape
    except ValueError:
        out = "ERRORE"
    print(s1, "+", s2, "->", out)

# %% [markdown]
# Uso pratico nel ML (è l'esercizio 5 del notebook 2.2): **centrare le feature**.
# Matrice dati `(n_campioni, n_feature)`, `data.mean(axis=0)` ha shape `(n_feature,)`,
# che per la regola 1 diventa `(1, n_feature)` e si stira su tutte le righe.
# Non ti scrivo la soluzione: provala tu.

# %% [markdown]
# **Esercizi del prof da fare ora:** `2.2_numpy_operations.ipynb`, esercizi 4 e 5.

# %% [markdown]
# ---
# ## 5. Accedere agli array
#
# | Tipo | Sintassi | View o copia? |
# |---|---|---|
# | Indicizzazione semplice | `x[1, 2]` | un singolo valore |
# | Slicing | `x[start:stop:step, ...]` | **VIEW** |
# | Masking | `x[x > 4]` | **COPIA** |
# | Fancy indexing | `x[[1, 3]]` | **COPIA** |
# | Combinata | `x[0, 1:]`, `x[[0, 2], :2]` | view solo se solo interi + slice, altrimenti copia |
#
# Questa colonna "view o copia" è il concetto più importante della sezione.

# %%
# Indicizzazione semplice: un intero per asse, separati da virgola
x = np.array([[2, 3, 4], [5, 6, 7]])
print(x[1, 2])   # lettura
x[1, 2] = 1      # scrittura
print(x)
print(x[0, -1], x[0, -2])  # indici negativi come nelle liste

# %%
# Slicing: start incluso, stop escluso; una slice per asse
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x[:, 1:])    # PREVEDI: tutte le righe, colonne dalla 1 in poi
print(x[:2, ::2])  # PREVEDI: prime 2 righe, colonne a passo 2
print(x[::-1])     # extra: righe al contrario

# %%
# UNA SLICE È UNA VIEW: scrivere nella view modifica l'originale
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
view = x[:, 1:]
view[:, :] = 0
print(x)  # x è cambiato!

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
safe = x[:, 1:].copy()
safe[:, :] = 0
print(x)  # x intatto

# Confronta con le liste Python: lista[1:] è una COPIA. NumPy si comporta al contrario.

# %%
# Meno indici che assi: i mancanti diventano ":" A DESTRA
x = np.array([[2, 3, 4], [5, 6, 7]])
print(x[1], x[1].shape)        # riga 1 = x[1, :]
print(x[:, 1], x[:, 1].shape)  # colonna 1: il ":" va scritto esplicitamente
print(x[:, 1:2], x[:, 1:2].shape)  # extra: con una slice l'asse NON sparisce -> (2, 1)
# Regola: un intero elimina il suo asse, una slice lo mantiene.

# %% [markdown]
# ### Masking
# - `x > 4` **non** dà un singolo True/False: dà un array di bool con la stessa shape di `x`.
# - `x[mask]` tiene gli elementi dove la maschera è True.
# - Con maschera della stessa shape di `x`, il risultato è **sempre 1-D** (gli
#   elementi scelti in generale non formano un rettangolo) ed è una **copia**.
# - Per combinare maschere: `&` (and), `|` (or), `^` (xor), `~` (not).
#   **Non** `and`/`or`, e **parentesi obbligatorie**: `&` e `|` hanno precedenza sui confronti.

# %%
x = np.array([1.2, 4.1, 1.5, 4.5])
mask = x > 4
print(mask)
print(x[mask])

x2 = np.array([[1.2, 4.1], [1.5, 4.5]])
print(x2[x2 > 1.3], x2[x2 > 1.3].shape)  # PREVEDI la shape: 1-D!

# %%
x = np.array([0.5, 1.0, 3.0, 5.0, 7.0])
print(x[(x >= 1) & (x <= 5)])
print(x[~((x < 1) | (x > 5))])  # stessa cosa

try:
    x[x >= 1 and x <= 5]  # sbagliato: 'and' su array
except ValueError as e:
    print("ERRORE con 'and':", e)

try:
    x[x >= 1 & x <= 5]  # sbagliato: mancano le parentesi
except Exception as e:
    print("ERRORE senza parentesi:", type(e).__name__)

# %%
# Scrivere ATTRAVERSO una maschera funziona; scrivere NEL risultato no
x = np.array([1.2, 4.1, 1.5, 4.5])
x[x > 4] = 0
print(x)  # modificato

x = np.array([1.2, 4.1, 1.5, 4.5])
masked = x[x > 4]  # copia
masked[:] = 0
print(x)  # intatto

# %% [markdown]
# ### Fancy indexing (per lista di indici)
# - **Una lista**: indicizza solo il primo asse, gli altri presi interi. `x[[1, 2]]` = righe 1 e 2.
# - **Due liste**: lette **a coppie**. `x[[1, 2], [0, 2]]` prende le celle (1,0) e (2,2),
#   **non** un blocco 2×2. Questo è l'errore più comune della sezione.
# - Anche qui: il risultato è una **copia**, ma scrivere attraverso l'indice funziona.

# %%
x = np.array([7.0, 9.0, 6.0, 5.0])
print(x[[1, 3]])

x2 = np.arange(9.0).reshape(3, 3)
print(x2)
print(x2[[1, 2]])              # righe intere
print(x2[[1, 2], [0, 2]])      # PREVEDI: coppie (1,0) e (2,2) -> 2 numeri
print(x2[[1, 2], :][:, [0, 2]])  # il BLOCCO righe {1,2} x colonne {0,2}
print(x2[np.ix_([1, 2], [0, 2])])  # extra: stesso blocco in un colpo solo

# %%
# Indicizzazione combinata: ogni intero "costa" un asse
x = np.arange(9.0).reshape(3, 3)
print(x[[True, False, True], 1:].shape)  # maschera + slice  -> 2D
print(x[[0, 2], :2].shape)               # fancy + slice     -> 2D
print(x[0, 1:].shape)                    # intero + slice    -> 1D
print(x[[True, False, True], 0].shape)   # maschera + intero -> 1D

# %% [markdown]
# **Esercizi del prof da fare ora:** `2.3_numpy_array_manipulation.ipynb`, esercizi 1 e 2.
# (Nel 2.2 del notebook ti chiede di modificare due colonne con una maschera:
# prima di eseguire, chiediti se stai scrivendo *attraverso* un'indicizzazione o
# *dentro* una copia.)

# %% [markdown]
# ---
# ## 6. Lavorare con gli array
#
# ### Concatenare lungo un asse ESISTENTE
# - `np.concatenate((x, y), axis=k)`: il risultato ha lo **stesso numero di dimensioni**.
# - Lungo l'asse `k` le dimensioni possono differire; **tutte le altre devono coincidere**.
# - `np.vstack` = axis 0 (uno sotto l'altro), `np.hstack` = axis 1 (affiancati) per le matrici.
# - Attenzione: gli array si passano come **tupla/lista**: `np.concatenate((x, y))`.

# %%
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[11, 12, 13], [14, 15, 16]])
print(np.concatenate((x, y)))          # default axis=0 -> (4, 3)
print(np.concatenate((x, y), axis=1))  # -> (2, 6)
print(np.vstack((x, y)).shape, np.hstack((x, y)).shape)

# %%
# vstack su vettori 1-D crea un asse NUOVO, concatenate no
x = np.array([1, 2, 3])
y = np.array([11, 12, 13])
print(np.vstack((x, y)))  # (2, 3)
try:
    np.concatenate((x, y), axis=1)
except Exception as e:
    print("ERRORE:", e)
print(np.concatenate((x, y)))  # questo va: (6,)
print(np.hstack((x, y)))       # PREVEDI: per vettori 1-D hstack = concatenate

# %% [markdown]
# ### Split
# `np.split(arr, N, axis=0)` restituisce una **lista** di array.
# - `N` intero → N parti **uguali** (errore se non è divisibile).
# - `N` lista → posizioni in cui **tagliare**.
# - `np.hsplit` = axis 1, `np.vsplit` = axis 0.

# %%
x = np.array([7, 7, 9, 9, 8, 8])
print(np.split(x, [2, 4]))  # taglia prima dell'elemento 2 e prima del 4
print(np.split(x, 3))       # 3 parti uguali: stesso risultato
try:
    np.split(x, 4)          # PREVEDI: 6 non è divisibile per 4
except ValueError as e:
    print("ERRORE:", e)

grid = np.array([[1, 2, 3, 11, 12, 13], [4, 5, 6, 14, 15, 16]])
left, right = np.hsplit(grid, [3])  # unpacking della lista restituita
print(left)
print(right)

# %% [markdown]
# ### Reshape
# - Riempie **riga per riga** (ordine "C").
# - Il numero totale di elementi deve restare uguale.
# - Al massimo **una** dimensione può essere `-1`: NumPy la calcola da solo.
# - `reshape(-1, 1)` → vettore colonna: lo userai di continuo con scikit-learn,
#   che vuole input 2D `(n_campioni, n_feature)`.

# %%
x = np.arange(6)
print(x.reshape((2, 3)))
print(x.reshape(3, -1))           # PREVEDI: -1 diventa?
print(np.array([1, 2, 3]).reshape(-1, 1))
try:
    x.reshape((4, 2))
except ValueError as e:
    print("ERRORE:", e)

# reshape di solito è una VIEW: stesso buffer letto in un altro modo
y = x.reshape(2, 3)
y[0, 0] = 99
print(x)  # PREVEDI: x cambia?

# %% [markdown]
# ### Salvare e caricare
# - `np.save("file", arr)` → `file.npy` (estensione aggiunta da sola); `np.load("file.npy")`.
# - `np.savez("archivio", x=a, y=b)` → `archivio.npz`, si legge come un dizionario.
# - `.npy` conserva dtype e shape esatti, a differenza di un CSV.

# %%
import os
import tempfile

d = tempfile.mkdtemp()
p = os.path.join(d, "tempfile")
x = np.arange(10)
np.save(p, x)
print(np.load(p + ".npy"))

np.savez(os.path.join(d, "archive"), x=x, y=np.arange(3))
archive = np.load(os.path.join(d, "archive.npz"))
print(archive.files, archive["y"])

# %% [markdown]
# **Esercizi del prof da fare ora:** `2.3_numpy_array_manipulation.ipynb`, esercizi 3 e 4.

# %% [markdown]
# ---
# ## Riassunto in 5 righe (dalla chiusura della lezione)
# 1. Un array è **un buffer contiguo con un solo dtype**: da qui memoria e velocità.
# 2. **Assi e shape** sono il vocabolario: un'aggregazione consuma l'asse che le dai,
#    un indice intero consuma l'asse che indicizzi.
# 3. **Broadcasting**: completa con 1 a sinistra, stira gli assi da 1, errore se due dimensioni > 1 differiscono.
# 4. **Slicing → view**, **masking e fancy → copie**: decide se l'assegnazione raggiunge l'originale.
# 5. **concatenate, split, reshape** riorganizzano gli array senza toccare i numeri.

# %% [markdown]
# ---
# ## Autoverifica (rispondi SENZA eseguire, poi controlla)
# Con `x = np.arange(12).reshape(3, 4)`:
# 1. `x.sum(axis=0).shape` ?
# 2. `x[1].shape` e `x[1:2].shape` ?
# 3. `(x + np.array([1, 2, 3])).shape` ?
# 4. `(x + np.array([[1], [2], [3]])).shape` ?
# 5. `x[x > 5].shape` ?
# 6. `x[[0, 2], [1, 3]]` quali valori?
# 7. Dopo `s = x[:, 0]; s[:] = -1`, `x[0, 0]` quanto vale? E con `s = x[:, [0]]`?
# 8. `np.argsort(np.array([3, 1, 2]))` ?
#
# Poi esegui la cella sotto per controllare.

# %%
x = np.arange(12).reshape(3, 4)
print(1, x.sum(axis=0).shape)
print(2, x[1].shape, x[1:2].shape)
try:
    print(3, (x + np.array([1, 2, 3])).shape)
except ValueError:
    print(3, "ERRORE")
print(4, (x + np.array([[1], [2], [3]])).shape)
print(5, x[x > 5].shape)
print(6, x[[0, 2], [1, 3]])
s = x[:, 0]
s[:] = -1
print(7, "view:", x[0, 0], end="   ")
x = np.arange(12).reshape(3, 4)
s = x[:, [0]]
s[:] = -1
print("fancy:", x[0, 0])
print(8, np.argsort(np.array([3, 1, 2])))
