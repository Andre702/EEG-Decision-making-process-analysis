import numpy as np
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

def compute_captum_analysis(model, test_loader, device, max_samples=200, sfreq=160.0):
    """
    Computes model attributions for given samples upt to the max_samples count using IntegratedGradients.
    Returns a dictionary containing results with appropriate tags for later analysis.
    """
    print("\n" + "=" * 60)
    print("Sample classification with captum neuron analysis (Computations)")
    print("=" * 60)

    model.eval()

    print("Calculating model accuracy on test set...")
    total_correct = 0
    total_samples = 0

    # before captum analysis, calculate accuracy on all the samples
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

    overall_accuracy = (total_correct / total_samples) * 100
    print(f"Accuracy: {overall_accuracy:.2f}% "
          f"({total_correct}/{total_samples} correct) <<<\n")
    ig = IntegratedGradients(model)

    dual_peak_samples = []
    all_samples_info = []
    processed = 0

    print("Calculating gradient integrals...")

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

            # Basic calculations along the axis
            time_imp_abs = np.mean(np.abs(attr_np), axis=0)
            signed_time_imp = np.mean(attr_np, axis=0)
            total_gradient_strength = np.sum(np.abs(attr_np))
            net_positive_force = np.sum(signed_time_imp)

            # Dual peaks analysis (this part searches for occurrences of 2 highly distinctive peaks
            # in neuron activation happening each before and after the midpoint of the recording
            visual_curve_magnitude = np.abs(signed_time_imp)
            time_length = len(visual_curve_magnitude)
            midpoint = time_length // 2

            max_first_half = np.max(visual_curve_magnitude[:midpoint])
            max_second_half = np.max(visual_curve_magnitude[midpoint:])

            global_max = max(max_first_half, max_second_half)
            smaller_max = min(max_first_half, max_second_half)

            if global_max > 1e-6 and smaller_max >= (0.5 * global_max):
                dual_peak_samples.append({
                    'class': target_class,
                    'pred_class': predicted_class,
                    'signed_imp': signed_time_imp,
                    'idx_1': np.argmax(visual_curve_magnitude[:midpoint]),
                    'idx_2': midpoint + np.argmax(visual_curve_magnitude[midpoint:]),
                })

            all_samples_info.append({
                'id': processed + 1,
                'true_class': target_class,
                'pred_class': predicted_class,
                'strength_abs': total_gradient_strength,
                'strength_net': net_positive_force,
                'time_imp_abs': time_imp_abs,
                'signed_imp': signed_time_imp,
                'heatmap': np.abs(attr_np),
                'raw_attr': attr_np,
            })

            processed += 1

        if processed >= max_samples:
            break

    if len(all_samples_info) == 0:
        print("No samples found.")
        return None

    # Processing the time axis (will be shared across all plots)
    n_channels, n_time = all_samples_info[0]['heatmap'].shape
    total_seconds = n_time / sfreq
    time_sec_array = np.linspace(0, total_seconds, n_time)

    # Data package with tags
    return {
        'all_samples_info': all_samples_info,
        'dual_peak_samples': dual_peak_samples,
        'time_sec_array': time_sec_array,
        'total_seconds': total_seconds,
        'n_channels': n_channels,
        'sfreq': sfreq
    }


def plot_dual_peaks(analysis_results, limit=6):
    """
    Generates classification proces plots for dual peaks
    - 2 highly distinctive peaks occurring before and after the midpoint of the recording.
    """
    if not analysis_results: return

    dual_peak_samples = analysis_results['dual_peak_samples']
    time_sec_array = analysis_results['time_sec_array']

    print("\n" + "=" * 60)
    print(f"Dual attention profiles. Found {len(dual_peak_samples)} samples")
    print("=" * 60)

    for drawn, anomaly in enumerate(dual_peak_samples):
        if drawn >= limit: break

        sig = anomaly['signed_imp']
        klasa = anomaly['class']
        klasa_pred = anomaly['pred_class']
        idx1, idx2 = anomaly['idx_1'], anomaly['idx_2']
        val1_signed, val2_signed = sig[idx1], sig[idx2]

        plt.figure(figsize=(10, 4))
        plt.plot(time_sec_array, sig, color='black', linewidth=1)
        plt.fill_between(time_sec_array, sig, 0, where=(sig >= 0), color='green', alpha=0.5)
        plt.fill_between(time_sec_array, sig, 0, where=(sig < 0), color='red', alpha=0.5)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)

        plt.scatter([time_sec_array[idx1]], [val1_signed], color='darkred' if val1_signed < 0 else 'darkgreen',
                    zorder=5, s=40, label=f'Peak A: {val1_signed:.3f}')
        plt.scatter([time_sec_array[idx2]], [val2_signed], color='darkred' if val2_signed < 0 else 'darkgreen',
                    zorder=5, s=40, label=f'Peak B: {val2_signed:.3f}')

        truth = 'Right Hand' if klasa == 1 else 'Left Hand'
        decision = 'Correct' if klasa == klasa_pred else 'INCORRECT!'

        plt.title(f"Captured dual profile - {truth} ({decision})")
        plt.xlabel("Event time [Seconds]")
        plt.ylabel("Network gradient strength")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # On some devices the plt waits here, on some it draws everything from a loop.
        # If the latter is observed it could be useful to wait for input here:
        # input("Press Enter...")


def plot_top_biased(analysis_results, top_n=3):
    """
    Generates time plots and heatmaps for samples with the strongest bias towards a single class
    either correct (always marked Green) or incorrect (always marked Red).
    """
    if not analysis_results: return

    all_samples_info = analysis_results['all_samples_info'].copy()  # Copy to prevent modifying other lists
    time_sec_array = analysis_results['time_sec_array']
    total_seconds = analysis_results['total_seconds']
    n_channels = analysis_results['n_channels']

    print("\n" + "=" * 60)
    print(f"Top {top_n} samples with the strongest single direction bias")
    print("=" * 60)

    # Sort by absolute sum (first sum, then abs)
    all_samples_info.sort(key=lambda x: abs(x['strength_net']), reverse=True)
    top_biased = all_samples_info[:top_n]

    for rank, sample in enumerate(top_biased, 1):
        true_name = 'Right Hand (1)' if sample['true_class'] == 1 else 'Left Hand (0)'
        pred_name = 'Right Hand (1)' if sample['pred_class'] == 1 else 'Left Hand (0)'
        correct_text = "Correct" if sample['true_class'] == sample['pred_class'] else "INCORRECT!"

        print(f"\n#{rank} | Sample ID: {sample['id']} | Classification result: {pred_name} -> {correct_text}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

        sig = sample['signed_imp']
        ax1.plot(time_sec_array, sig, color='black', linewidth=1)
        ax1.fill_between(time_sec_array, sig, 0, where=(sig >= 0), color='limegreen', alpha=0.6, label="Towards class 1")
        ax1.fill_between(time_sec_array, sig, 0, where=(sig < 0), color='red', alpha=0.6, label="Towards class 0")
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.set_title(f"Strong directional gradient distribution (Truth (Green): {true_name})")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Directional gradient")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        # Heatmap
        im = ax2.imshow(sample['heatmap'], aspect='auto', cmap='hot', origin='lower',
                        extent=[0, total_seconds, 0, n_channels])
        ax2.set_title("Heatmap")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Electrode number")
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        plt.show()


def plot_top_conflicted(analysis_results, top_n=3):
    """
    Finds samples where the network swayed strongly and equally towards both classifications
    (where the ultimate decision was conflicted and influenced by both features).
    """
    if not analysis_results: return

    all_samples_info = analysis_results['all_samples_info'].copy()
    time_sec_array = analysis_results['time_sec_array']
    total_seconds = analysis_results['total_seconds']
    n_channels = analysis_results['n_channels']

    print("\n" + "=" * 60)
    print(f"Top {top_n} most decision-conflicted samples")
    print("=" * 60)

    for s in all_samples_info:
        sig = s['signed_imp']
        # sums gradients towards both classes separately
        pos_sum = np.sum(sig[sig > 0]) if len(sig[sig > 0]) > 0 else 0
        neg_sum = np.abs(np.sum(sig[sig < 0])) if len(sig[sig < 0]) > 0 else 0
        # The actual sample is chosen based on the strength of the weaker class
        s['conflict_score'] = min(pos_sum, neg_sum)

    all_samples_info.sort(key=lambda x: x['conflict_score'], reverse=True)
    top_conflicted = all_samples_info[:top_n]

    for rank, sample in enumerate(top_conflicted, 1):
        true_name = 'Right Hand (1)' if sample['true_class'] == 1 else 'Left Hand (0)'
        pred_name = 'Right Hand (1)' if sample['pred_class'] == 1 else 'Left Hand (0)'
        correct_text = "Correct" if sample['true_class'] == sample['pred_class'] else "INCORRECT!"

        print(f"\n#{rank} | Sample ID: {sample['id']} | Classification: {pred_name} -> {correct_text}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

        sig = sample['signed_imp']
        ax1.plot(time_sec_array, sig, color='black', linewidth=1)
        ax1.fill_between(time_sec_array, sig, 0, where=(sig >= 0), color='green', alpha=0.5, label='+ do klasy 1')
        ax1.fill_between(time_sec_array, sig, 0, where=(sig < 0), color='red', alpha=0.5, label='- do klasy 0')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.set_title(f"Strong decision conflict in classification (Truth (green): {true_name})")
        ax1.set_xlabel("Czas [s]")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        im = ax2.imshow(sample['heatmap'], aspect='auto', cmap='hot', origin='lower',
                        extent=[0, total_seconds, 0, n_channels])
        ax2.set_title("Heatmap")
        ax2.set_xlabel("Time [s]")
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        plt.show()


def extract_global_heatmap_data(analysis_results, mode='all'):
    """
    Extracts and averages the (channels x time) attribution matrix across all samples.

    Modes:
    - 'all': Absolute sum of all attributions.
    - 'correct_direction': Only features that pushed the model towards the true class (attr > 0).
    - 'incorrect_direction': Only features that pushed the model towards the opposing class (attr < 0).
    """
    if not analysis_results or 'all_samples_info' not in analysis_results:
        return None

    all_samples = analysis_results['all_samples_info']
    n_channels = analysis_results['n_channels']
    n_time = len(analysis_results['time_sec_array'])

    accumulated_heatmap = np.zeros((n_channels, n_time))
    valid_samples_count = len(all_samples)

    if valid_samples_count == 0:
        return None

    for s in all_samples:
        raw_attr = s['raw_attr']

        if mode == 'all':
            filtered_attr = np.abs(raw_attr)

        elif mode == 'correct_direction':
            # Keeps only positive attributions (towards the true target class)
            filtered_attr = np.where(raw_attr > 0, raw_attr, 0)

        elif mode == 'incorrect_direction':
            # Keeps only negative attributions (away from the true target class) as absolute value
            filtered_attr = np.abs(np.where(raw_attr < 0, raw_attr, 0))

        else:
            raise ValueError("Unknown mode. Use 'all', 'correct_direction', or 'incorrect_direction'.")

        accumulated_heatmap += filtered_attr

    # Return the average heatmap across all processed samples
    return accumulated_heatmap / valid_samples_count


def plot_global_heatmap_and_bars(heatmap_data, analysis_results, title_suffix="All Attributions"):
    """
    Plots an aggregated heatmap from all the samples as well as a horizontal bar chart
    representing total absolute attention per channel without time parameter.
    """
    if heatmap_data is None or analysis_results is None:
        return

    time_sec_array = analysis_results['time_sec_array']
    total_seconds = analysis_results['total_seconds']
    n_channels = analysis_results['n_channels']

    print("\n" + "=" * 60)
    print(f"Generating Global Feature Distribution - {title_suffix}")
    print("=" * 60)

    # Setup figure with 1 row, 2 columns, specific width ratios (3:1)
    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1]})

    im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='hot', origin='lower',
                        extent=[0, total_seconds, 0, n_channels])

    ax_heat.set_title(f"Average Global Attention Heatmap ({title_suffix})")
    ax_heat.set_xlabel("Time [s]")
    ax_heat.set_ylabel("Channel (Electrode)")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # Collapse the time dimension by summing attributions across each channel
    channel_importance = np.sum(heatmap_data, axis=1)

    y_pos = np.arange(n_channels) + 0.5

    ax_bar.barh(y_pos, channel_importance, align='center', color='royalblue', edgecolor='black')

    ax_bar.set_ylim(0, n_channels)
    # Hide y-ticks on the bar chart since they align with the heatmap
    ax_bar.set_yticks([])

    ax_bar.set_title("Total Channel Impact")
    ax_bar.set_xlabel("Accumulated Gradient Strength")
    ax_bar.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()