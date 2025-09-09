import gurobipy as gp
from gurobipy import GRB

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utilis.airport_test_set_init import make_new_test_set

airports_df = make_new_test_set()

try:
    plt.figure(figsize=(10, 8))

    plt.scatter(airports_df['x'], airports_df['y'], label='Aeroporti')

    # --- Aggiunta delle Etichette per ogni Aeroporto ---
    # Aggiunge un'etichetta con l'ID accanto a ogni punto per identificarlo
    for i, row in airports_df.iterrows():
        plt.text(row['x'] + 5, row['y'] + 5, str(row['airport_id']))

    plt.title('Mappa degli Aeroporti Generati')
    plt.xlabel('Coordinata X (km)')
    plt.ylabel('Coordinata Y (km)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

except NameError:
    print("Errore: la variabile 'airports_df' non è definita.")
    print("Assicurati di eseguire prima la cella che crea o carica i dati.")
except Exception as e:
    print(f"Si è verificato un errore durante la creazione del grafico: {e}")