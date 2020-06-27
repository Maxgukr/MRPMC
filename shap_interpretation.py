import shap
import matplotlib.pyplot as plt


def run_shap(results, X_test, new_path, hp):
    for key, item in results.items():
        model = item[0]
        if key == 'EmsembleModel':
            continue
        X_test = X_test[:10, :]
        k_X = shap.kmeans(X_test, 5)
        explainer = shap.KernelExplainer(model.model.predict, k_X)
        shap_values = explainer.shap_values(X_test)
        f = plt.figure()
        shap.summary_plot(shap_values, X_test)
        f.savefig(new_path + "/" + hp + '-' + key + "-summary_plot.pdf", bbox_inches='tight', dpi=600)
        f = plt.figure()
        shap.force_plot(explainer.expected_value[0], shap_values, X_test)
        f.savefig(new_path + "/" + hp + '-' + key + "-force_plot.pdf", bbox_inches='tight', dpi=600)