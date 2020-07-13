import shap
import matplotlib.pyplot as plt


def run_shap(results, X_test, new_path, hp):
    for key, item in results.items():
        model = item[0]
        # X_test = X_test.iloc[:10, :]
        k_X = shap.kmeans(X_test, 5)
        if key == 'MLP':
            explainer = shap.KernelExplainer(model.model.predict_classes, k_X)
        else:
            explainer = shap.KernelExplainer(model.model.predict, k_X)
        shap_values = explainer.shap_values(X_test)
        f = plt.figure()
        shap.summary_plot(shap_values, X_test, show=False, plot_type='bar')
        f.savefig(new_path + "/" + hp + '-' + key + "-summary_plot.pdf", bbox_inches='tight', dpi=600)
