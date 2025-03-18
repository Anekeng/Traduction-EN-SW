import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
import datasets
import torchtext
import tqdm
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


donnees = pd.read_csv("swahili_english_translation.csv")
df = donnees.head(50000)
df_p = pd.DataFrame(columns=["en", "ye"])
df_p["en"] = df.apply(lambda row: row["input"] if row.name % 2 == 0 else row["output"], axis=1)
df_p["ye"] = df.apply(lambda row: row["output"] if row.name % 2 == 0 else row["input"], axis=1)


##df_p.describe()
##df_p
# Convertir le DataFrame en une liste de dictionnaires
data_list = df_p.to_dict(orient='records')
# Séparer les données en ensembles d'entraînement, validation et test
train, test = train_test_split(data_list, test_size=0.2, random_state=42)  # 80% train, 20% test
train, validation = train_test_split(train, test_size=0.2, random_state=42)  # 80% train, 20% validation


# Afficher les tailles des ensembles
print(f"Taille de l'échantillon d'entraînement: {len(train)}")
print(f"Taille de l'échantillon de validation: {len(validation)}")
print(f"Taille de l'échantillon de test: {len(test)}")
train_dataset = Dataset.from_list(train)
validation_dataset = Dataset.from_list(validation)
test_dataset = Dataset.from_list(test)
# Construire le DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})


# Vérification de la structure
##print(dataset)
##corpus = df_p["ye"]
##pd.DataFrame(corpus).to_csv('corpus.txt', index=False)
##corpus2 = df_p["en"]
##pd.DataFrame(corpus2).to_csv('corpus2.txt', index=False)


# Tester le tokenizer
#encoded = tokenizer.encode("Mtu aliyepanda farasi anaruka juu ya ndege iliyovunjika.")
#print(encoded.tokens)  # Affiche la tokenization

ye_nlp = Tokenizer.from_file("TokenizerSW.json")
en_nlp = Tokenizer.from_file("TokenizerEN.json")
train_data, validation_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)


def yield_tokens(dataset, lang):
    """Générateur de tokens pour le vocabulaire."""
    for example in dataset:
        yield example[lang].split()  # Tokenisation naïve par espaces


# Construire le vocabulaire pour chaque langue
unk_token = "<unk>"
pad_token = "<pad>"
bos_token = "<bos>"
eos_token = "<eos>"
en_vocab = build_vocab_from_iterator(yield_tokens(train_data, "en"), specials= [unk_token, pad_token, bos_token, eos_token])
ye_vocab = build_vocab_from_iterator(yield_tokens(train_data, "ye"), specials= [unk_token, pad_token, bos_token, eos_token])

# Configurer l'index spécial pour <unk>
en_vocab.set_default_index(en_vocab["<unk>"])
ye_vocab.set_default_index(ye_vocab["<unk>"])
# Vérification
#print("Taille du vocabulaire Anglais:", len(en_vocab))
#print("Taille du vocabulaire Yemba:", len(ye_vocab))


# Exemple d'indexation de mots
#print("Index de 'the':", en_vocab["the"])  # Retourne l'index si le mot existe, sinon <unk>
#print("Index de 'anaruka':", ye_vocab["anaruka"])  # Mot en swalli
#en_vocab.get_itos()[:10]
#ye_vocab.get_itos()[:10]
assert ye_vocab[unk_token] == en_vocab[unk_token]
assert ye_vocab[pad_token] == en_vocab[pad_token]


unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]
en_vocab.set_default_index(unk_index)
ye_vocab.set_default_index(unk_index)
tokens = ["our", "riparian", "field", "is", "near", "the", "bridge"]
en_vocab.lookup_indices(tokens)

# Fonction pour convertir une phrase en indices
def numericalize(sentence, vocab):
    return [vocab[bos_token]] + [vocab[token] for token in sentence.split()] + [vocab[eos_token]]


# Fonction de traitement d'un lot
def collate_fn(batch):
    en_batch = [torch.tensor(numericalize(item["en"], en_vocab)) for item in batch]
    ye_batch = [torch.tensor(numericalize(item["ye"], ye_vocab)) for item in batch]
    en_batch = pad_sequence(en_batch, padding_value=en_vocab[pad_token])
    ye_batch = pad_sequence(ye_batch, padding_value=ye_vocab[pad_token])
    return {"en": en_batch, "ye": ye_batch}


# Création des chargeurs de données
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(validation_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
print("Chargeurs de données prêts!")


# Définition de l'encodeur
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


# Définition du décodeur
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell
    

# Définition du modèle Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0, :]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        
        return outputs
    

# Initialisation des hyperparamètres
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(en_vocab)
output_dim = len(ye_vocab)
emb_dim = 256
hid_dim = 512
n_layers = 2
dropout = 0.5


# Initialisation du modèle
encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout).to(device)
decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout).to(device)

# Chargement du modèle sauvegardé
def load_model(model, optimizer, model_path="saved_models/best_model.pth"):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Modèle chargé depuis l'époque {epoch} avec une perte de validation de {loss:.3f}")
    return model, optimizer

# Initialisation des hyperparamètres
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(en_vocab)
output_dim = len(ye_vocab)
emb_dim = 256
hid_dim = 512
n_layers = 2
dropout = 0.5

# Initialisation du modèle
encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout).to(device)
decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout).to(device)
model = Seq2Seq(encoder, decoder, device).to(device) 
optimizer = optim.Adam(model.parameters())
model_path="saved_models/best_model.pth"
model, optimizer = load_model(model, optimizer, model_path)

def translate_sentence(sentence, model, en_vocab, ye_vocab, device, max_length=50):
    """Traduit une phrase de l'anglais vers le yemba."""
    model.eval()
    
    # Tokenisation et conversion en indices
    tokens = sentence.lower().split()
    numericalized = [en_vocab["<bos>"]]
    
    # Ajouter chaque token avec un fallback sur <unk> si le mot n'existe pas
    for token in tokens:
        try:
            # Tentative d'accès direct au vocabulaire
            numericalized.append(en_vocab[token])
        except KeyError:
            # Si le mot n'est pas trouvé, ajouter <unk>
            numericalized.append(en_vocab["<unk>"])
    
    numericalized.append(en_vocab["<eos>"])
    
    # Conversion en tenseur PyTorch
    src_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)
    
    # Passage dans l'encodeur
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
    
    # Décodage itératif
    trg_indexes = [ye_vocab["<bos>"]]
    
    for _ in range(max_length):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == ye_vocab["<eos>"]:
            break
    
    # Conversion des indices en mots
    trg_tokens = [ye_vocab.lookup_token(idx) for idx in trg_indexes]
    
    return " ".join(trg_tokens[1:-1])  # Exclure <bos> et <eos>


# Interface utilisateur avec Streamlit
import streamlit as st
st.title("Chatbot de Traduction Anglais → Swahili")

phrases = test_data['en'][:30]
traduction = test_data['ye'][:30]
df_phrases = pd.DataFrame({"Phrases en anglais": phrases, "Phrases en swahili": traduction})
df_phrase = df_phrases.drop_duplicates()
df_phrase

# Affichage du tableau
st.subheader("Phrases suggérées à traduire")
st.table(df_phrase)
st.write("Entrez une phrase en anglais et obtenez la traduction en Swahili.")

# Saisie utilisateur
sentence = st.text_input("Entrez votre texte en anglais :", "")

if st.button("Traduire"):
    if sentence:
        translation = translate_sentence(sentence, model, en_vocab, ye_vocab, device)
        st.success(f"**Traduction en Swahili :** {translation}")
    else:
        st.warning("Veuillez entrer une phrase en anglais.")