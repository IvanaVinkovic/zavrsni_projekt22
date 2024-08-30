# Redukcija višedimenzionalnih podataka

## Opis
### Redukcija višedimenzionalnih podatak je proces obrade višedimenziolnih podataka na način da se smanji broj ulaznih varijabli odnosno dimenzija, a zadražvaju se ključne informacije u podacima. To je posebno korisno pri radu s veliki dimenzijama podataka i puno atributa kako bi se omogućila što bolja vizualizacija i primjena algoritama iz smjera strojnog učenja. Kako bismo to postigli koristili smo linearne mappere poput PCA i SVD te nelinearne mapper kao što su mMDS i Autoencoder koje smo konstruirali pomoću neruonskih mreža.

## Sadržaj
  * Instalacija 
  * PCA
  * SVD
  * mMDS
  * Autoencoder

## Instalacija
Da biste instalirali i pokrenuli ovaj projekt na svom računali slijedite ove korake:
  * klonirajte ovaj repozitorij na svoje računalo https://github.com/IvanaVinkovic/zavrsni_projekt22.git
  * cd završni_projekt22/zp22
  * (opcionalno) možete postaviti novo okruženje kako biste izolirali ovisnosti projekta:
     * python3 -m venv env
     * source env/bin/activate   (Na Windows sustavu koristite `env\Scripts\activate)
  * instalirajte potrebne pakete koristeći: pip -r install requirements.txt
  * aplikaciju pokrenite u terminalu pomoću naredbe pyhon app.py

## PCA
PCA je tehnika za redukciju dimenzionalnosti koja identificira skup ortogonalnih osi koje zovemo glavne komponente. One obuhvaćaju maksimalnu varijancu u podacima. Glavne komponente su linearne kombinacije originalnih varijabli u skupu podataka i poredane su prema važnosti u padajućem redoslijedu. Ukupna varijanca obuhvaćena svim glavnim komponentama jednaka je ukupnoj varijanci u izvornom skupu podataka. Matematički PCA se opisuje na sljedeći način:

1. korak: Standarizacija

$$Z = \frac{X - \mu}{\sigma}$$


  *    **$\mu$** - srednja vrijednost neovisnih značajki {**$\mu_1$**,**$\mu_2$**,...,**$\mu_m$**}
          
  *    **$\sigma$** - standardna devijacija neovisnih značajki {**$\sigma_1$**,**$\sigma_2$**,...,**$\sigma_m$**}

2. korak: Izračun kovarijacijske matrice

$$cov(x_1, x_2) = \frac{\sum_{i=1}^n (x_{1_i} - \overline{x_1}) (x_{2_i} - \overline{x_2})}{n - 1}$$

  vrijednost varijance može biti pozitivna, negativna ili nula
  
  * Pozitivna: ako se poveća $x_1$ poveća se i $x_2$
  * Negativna: ako se smanji $x_1$ smanji se i $x_2$
  * Nula: nema direktne povezanosti

3. Svojstvene vrijednosti i svojstveni vektor

   Neka je A matrica nxn i neka je X nenul vektor za koji vrijedi $$AX = \lambda \cdot X$$, $\lambda$ je svojstvena vrijednost matrice A, a X je njen svojstveni vektor. Može se pisati:

   $$AX - \lambda \cdot X = 0$$
   
   $$(A - \lambda \cdot I) \cdot X = 0$$

## SVD
Neka je $A \in \mathbb{R}^{m,n}$. Jedinični vektor $v \in \mathbb{R}^{n}$ i $u \in \mathbb{R}^{m}$ su lijevi i desni singularni vektori matrice A za odgovarajućom vrijednosti $\sigma > 0$ ako je $AV = \sigma \cdot u$ i $A^T u = \sigma \cdot v$. Ako možemo pronaći r ortogonalni singularni vektor sa pozitivnim singularnim vrijednostima, on možemo napraviti dekompoziciju $A = UDV^T$, stupci U i V matrica sadrže lijeve i desne singularne vrijednosti, a D je dijagonalna matrica rxr sa singularnim vrijednostima na svojoj dijagonali. Matematički:

$$ A = U \cdot \Sigma \cdot V^T$$

  * U: mxm matrica ortogonalnih sovjstvenih vektora od $AA^T$
  * $V^T$: transponirana matrica nxn koja sadrži ortonormirane svojstvene vektore od $A^TA$
  * $\Sigma$: dijagonalna matrica sa r elemenata koji su jednaki pozitivnim svojstvenim vrijednostima $AA^T$ ili $AA^T$   (obje matrice imaju jednake svojstvene vrijednosti)

## metrical MDS
  1. klasa MDSNet: Predstavlja neuralnu mrežu implementiranu u TensorFlowu koristeći KeraS API. Ova mreža sadrži slojeve gustoće i slojeve za regularizaciju i dropout. Koristi se za smanjenje dimenzionalnosti i za regresiju
  2. funkcija pairwise_distance_ funkcija koristi matricu skalarnih proizvoda za izračun udaljenosti između redaka. Prvo se izračunava skalarni proizvod između redaka, zatim se koristi za izračun kvadratnih normi i konačno se dobivaju udaljenosti koristeći kvadratne korijene. Koriste se numerički stabilizatori kako bi se osiguralo da udaljenosti ne posatnu negativne zbog numeričkih pogrešaka

## Autoencoder 
  Autoencoder je neuronska mreža implementirana pomoću TensorFlow-a i Keras-a. Ona uči kodiranje ulaznih podataka u komprimirani oblik i njihovu rekonstrukciju. Ulazni model prima podatke, kodirajući slojevi smanjuju dimenzionalnost kroz slojeve gustoće s funkcijama aktivacije 'relu' i 'linear', dekodirajući slojevi obnavljaju dimenzionalnost na izvornu veličinu koristeći slojeve gustoće s funkcijama aktivacije 'relu' i 'sigmoid'. Model se kompajlira koristeći Adam optimizator i funkciju gubitka 'mae' (Mean Absolute Error). Model se trenira na 'data_reduced' tokom 500 epoha s veličinom serije od 512. Koristi se kodirajući dio modela za predviđanje smanjenih dimenzija ulaznih podataka.

### Napravili: Ivana Vinković, Juraj Mesarić
### Mentor: Domagoj Matijević


