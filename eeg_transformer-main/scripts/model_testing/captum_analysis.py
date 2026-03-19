import numpy as np
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# Interpretacja Captum i wyniki ==============================================================================

def analyze_bulk(model, test_loader, device, max_samples=200):
    print("\n" + "=" * 60)
    print(f"Analiza klasyfikacji próbek przez Captum z testowego setu")
    print("=" * 60)

    model.eval()
    ig = IntegratedGradients(model)

    attr_class_0 = []
    attr_class_1 = []

    dual_peak_samples = []
    all_samples_info = []

    processed = 0
    print("Obliczanie całek gradientów")

    for X_batch, y_batch in test_loader:
        for i in range(len(X_batch)):
            if processed >= max_samples:
                break

            X_input = X_batch[i].unsqueeze(0).to(device).requires_grad_()
            target_class = y_batch[i].item()

            output = model(X_input)
            predicted_class = torch.argmax(output, dim=1).item()

            attributions, _ = ig.attribute(X_input, target=target_class, return_convergence_delta=True, n_steps=20)
            attr_np = attributions.cpu().detach().numpy().squeeze()

            # Oś z samymi wartościami absolutnymi
            time_imp_abs = np.mean(np.abs(attr_np), axis=0)

            signed_time_imp = np.mean(attr_np, axis=0)

            total_gradient_strength = np.sum(np.abs(attr_np))
            net_positive_force = np.sum(signed_time_imp)

            all_samples_info.append({
                'strength_abs': total_gradient_strength,
                'strength_net': net_positive_force,
                'time_imp_abs': time_imp_abs,
                'signed_imp': signed_time_imp,
                'heatmap': np.abs(attr_np),
                'true_class': target_class,
                'pred_class': predicted_class,
                'id': processed + 1
            })

            visual_curve_magnitude = np.abs(signed_time_imp)

            time_length = len(visual_curve_magnitude)
            midpoint = time_length // 2

            max_first_half = np.max(visual_curve_magnitude[:midpoint])
            max_second_half = np.max(visual_curve_magnitude[midpoint:])

            global_max = max(max_first_half, max_second_half)
            smaller_max = min(max_first_half, max_second_half)

            if global_max > 1e-6:
                if smaller_max >= (0.5 * global_max):
                    dual_peak_samples.append({
                        'class': target_class,
                        'pred_class': predicted_class,
                        'signed_imp': signed_time_imp,
                        'idx_1': np.argmax(visual_curve_magnitude[:midpoint]),
                        'idx_2': midpoint + np.argmax(visual_curve_magnitude[midpoint:]),
                    })

            # Klasyfikacja do średniej ogólnej (bezwzględna) - aby zapisać bezwzględną chwilową uwagę sieci
            if target_class == 0:
                attr_class_0.append(np.abs(attr_np))
            else:
                attr_class_1.append(np.abs(attr_np))

            processed += 1
            # Test capowany na maksymalnej liczbie próbek. Można będzie zwiększyć przy dużym zbiorze

        if processed >= max_samples:
            break

    count_0 = len(attr_class_0)
    count_1 = len(attr_class_1)
    total_valid = count_0 + count_1

    if total_valid == 0: return

    # Oś Czasu
    n_time = attr_class_0[0].shape[1] if count_0 > 0 else attr_class_1[0].shape[1]
    n_channels = attr_class_0[0].shape[0] if count_0 > 0 else attr_class_1[0].shape[0]

    SFREQ = 160.0
    total_seconds = n_time / SFREQ
    time_sec_array = np.linspace(0, total_seconds, n_time)

    # WYKRESY =================================================================

    # Wykresy podwójnych profili uwagi ---------------------------------------------------------------------
    # Są to wykresy przebiegu klasyfikacji, w której potencjalnie można zauważyć skupienie sieci na dwóch różnych miejscach.
    # Jedno przed drugie po trzeciej sekundzie nagrania

    print("\n" + "=" * 60)
    print(f"Podwójne profile uwagi. Znaleziono {len(dual_peak_samples)} podwójnych skoków (przed i po 3. sekundzie)")
    print("=" * 60)

    limit_draw = 6  #graficzne przedstawienie 6 piewszych
    drawn = 0

    for anomaly in dual_peak_samples:
        if drawn >= limit_draw: break

        sig = anomaly['signed_imp']
        klasa = anomaly['class']
        klasa_pred = anomaly['pred_class']

        # Odczytywane wartości ze znakiem na szczytowych skokach
        idx1, idx2 = anomaly['idx_1'], anomaly['idx_2']
        val1_signed = sig[idx1]
        val2_signed = sig[idx2]

        plt.figure(figsize=(10, 4))

        plt.plot(time_sec_array, sig, color='black', linewidth=1)
        plt.fill_between(time_sec_array, sig, 0, where=(sig >= 0), color='green', alpha=0.5)
        plt.fill_between(time_sec_array, sig, 0, where=(sig < 0), color='red', alpha=0.5)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)

        plt.scatter([time_sec_array[idx1]], [val1_signed], color='darkred' if val1_signed < 0 else 'darkgreen',
                    zorder=5, s=40, label=f'Peak A: {val1_signed:.3f}')
        plt.scatter([time_sec_array[idx2]], [val2_signed], color='darkred' if val2_signed < 0 else 'darkgreen',
                    zorder=5, s=40, label=f'Peak B: {val2_signed:.3f}')

        prawda = 'Prawa Ręka' if klasa == 1 else 'Lewa Ręka'
        wyrok = 'Klasyfikacja poprawna' if klasa == klasa_pred else 'Klasyfikacja BŁĘDNA!'

        plt.title(f"Uchwycony profil podwójny - {prawda} {wyrok}")
        plt.xlabel("Moment zdarzenia [Sekundy]")
        plt.ylabel("Siła gradientu sieci")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        drawn += 1

        input("...")

    # Wykresy Top 3 najlepiej ukierunkowanych próbek ---------------------------------------------------------------------------------
    # Są to wykresy próbek najprostszych dla sieci do sklasyfikowania. Czyli tych, które miały największy bias do jednej konkretnej klasy.
    # Nie oznacza to jednak, że sieć faktycznie poprawnie sklasyfikowała próbkę.

    print("\n" + "=" * 60)
    print("Top 3 próbek pod względem pewność absolutnej jednej z klas")
    print("=" * 60)

    # Sortowanie po absolutnej sumie kierunkowej. |sum(+ i -)|
    all_samples_info.sort(key=lambda x: abs(x['strength_net']), reverse=True)
    top_3_biased = all_samples_info[:3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    for rank, sample in enumerate(top_3_biased, 1):
        true_name = 'Prawa Ręka (1)' if sample['true_class'] == 1 else 'Lewa Ręka (0)'
        pred_name = 'Prawa Ręka (1)' if sample['pred_class'] == 1 else 'Lewa Ręka (0)'
        correct_text = "Poprawna" if sample['true_class'] == sample['pred_class'] else "BŁĘDNA"

        print(f"\nPróbka #{rank} | ID Próbki: {sample['id']} | Siła gradientu: {sample['strength_net']:.2f}")
        print(f"Wynik klasyfikacji: {pred_name} -> Klasyfikacja {correct_text}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

        sig = sample['signed_imp']
        ax1.plot(time_sec_array, sig, color='black', linewidth=1)
        ax1.fill_between(time_sec_array, sig, 0, where=(sig >= 0), color='limegreen', alpha=0.6,
                         label="Kierunek poprawny")
        ax1.fill_between(time_sec_array, sig, 0, where=(sig < 0), color='red', alpha=0.6, label="Kierunek błędny")
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)

        ax1.set_title(f"#{rank} Rozkład gradientu sieci silnie ukierunkowanej (Próbka: {true_name})")
        ax1.set_xlabel("Moment zdarzenia [Sekundy]")
        ax1.set_ylabel("Siła gradientu sieci")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        # Heatmap
        im = ax2.imshow(sample['heatmap'], aspect='auto', cmap='hot', origin='lower',
                        extent=[0, total_seconds, 0, n_channels])
        ax2.set_title("Heatmapa")
        ax2.set_xlabel("Moment zdarzenia [Sekundy]")
        ax2.set_ylabel("Kanały (Elektrody)")
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        plt.show()
        input("...")

    # Wykresy Top 3 próbki najbardziej nieukierunkowanych ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Top 3 próbek pod względem różnicy gradientów obu klas")
    print("=" * 60)

    # Obliczenia największych dwóch przeciwnych gradientów
    for s in all_samples_info:
        sig = s['signed_imp']
        # suma dodatnich aktywacji
        pos_sum = np.sum(sig[sig > 0]) if len(sig[sig > 0]) > 0 else 0
        # suma absolutna ujemnych aktywacji
        neg_sum = np.abs(np.sum(sig[sig < 0])) if len(sig[sig < 0]) > 0 else 0

        # Wynik to wartość słabszego z gradientów. Im wyższa wartość minimalna, tym większe były różnice
        s['conflict_score'] = min(pos_sum, neg_sum)

    # Ponowne sortowanie pod względem conflict_score
    all_samples_info.sort(key=lambda x: x['conflict_score'], reverse=True)
    top_3_conflicted = all_samples_info[:3]

    for rank, sample in enumerate(top_3_conflicted, 1):
        true_name = 'Prawa Ręka (1)' if sample['true_class'] == 1 else 'Lewa Ręka (0)'
        pred_name = 'Prawa Ręka (1)' if sample['pred_class'] == 1 else 'Lewa Ręka (0)'
        correct_text = "Poprawna" if sample['true_class'] == sample['pred_class'] else "BŁĘDNA"

        # Dodatkowe wyliczenie sił do analizy
        s_sig = sample['signed_imp']
        P_val = np.sum(s_sig[s_sig > 0])
        N_val = np.sum(s_sig[s_sig < 0])

        print(f"\nPróbka #{rank} | ID Próbki: {sample['id']}")
        print(f"Gradient ku klasie poprawnej {P_val:.2f} | Gradient ku klasie błędnej {N_val:.2f}")
        print(f"Wynik klasyfikacji: {pred_name} -> Klasyfikacja {correct_text}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

        sig = sample['signed_imp']
        ax1.plot(time_sec_array, sig, color='black', linewidth=1)
        ax1.fill_between(time_sec_array, sig, 0, where=(sig >= 0), color='green', alpha=0.5,
                         label='Kierunek poprawny')
        ax1.fill_between(time_sec_array, sig, 0, where=(sig < 0), color='red', alpha=0.5,
                         label='Kierunek błędny')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)

        ax1.set_title(f"#{rank} Rozkład gradientu sieci sieci silnie nieukierunkowanej (Próbka: {true_name})")
        ax1.set_xlabel("Moment zdarzenia [Sekundy]")
        ax1.set_ylabel("Siła gradientu sieci")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        im = ax2.imshow(sample['heatmap'], aspect='auto', cmap='hot', origin='lower',
                        extent=[0, total_seconds, 0, n_channels])
        ax2.set_title("Heatmapa")
        ax2.set_xlabel("Moment zdarzenia [Sekundy]")
        ax2.set_ylabel("Kanały (Elektrody)")
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        plt.show()
        input("...")
