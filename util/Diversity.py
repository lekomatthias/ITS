import numpy as np
from collections import Counter

class Diversity:
    def __init__(self, segments=None):
        self.segments = segments
        if segments is not None and len(segments) > 0: 
            self.species_data = self._auto_generate_species_dict(segments)
        else:
            self.species_data = {}

    def _auto_generate_species_dict(self, segments):
        flat_segments = segments.flatten()
        segment_counts = Counter(flat_segments)
        id_to_species = {seg_id: f'espécie_{i+1}' for i, seg_id in enumerate(sorted(segment_counts))}
        species_named = [id_to_species[seg_id] for seg_id in flat_segments]
        species_data = dict(Counter(species_named))

        return species_data

    def richness(self):
        """
        Calcula a riqueza de espécies.
        """
        return len(self.species_data)

    def simpson_index(self):
        """
        Calcula o índice de diversidade de Simpson (1 - D).
        """
        values = list(self.species_data.values())
        N = sum(values)
        if N <= 1:
            return 0.0
        return 1 - (sum(n * (n - 1) for n in values) / (N * (N - 1)))

    def shannon_index(self):
        """
        Calcula o índice de Shannon-Wiener (H').
        """
        values = list(self.species_data.values())
        N = sum(values)
        if N == 0:
            return 0.0
        H = 0
        for n in values:
            p_i = n / N
            if p_i > 0:
                H -= p_i * np.log2(p_i)
        return H

    def summary(self):
        """
        Exibe os resultados dos três índices.
        """
        print(f"Número de espécies: {self.richness()}")
        print(f"Simpson: {round(self.simpson_index(), 2)}")
        print(f"Shannon-Wiener: {round(self.shannon_index(), 2)}")


if __name__ == "__main__":
    from tkinter import filedialog
    segments = np.load(filedialog.askopenfilename())
    div = Diversity(segments)
    div.summary()
