from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def entrenar_y_mostrar(max_depth):
    wine = load_wine()
    X, y = wine.data, wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    precision = accuracy_score(y_test, y_pred)

    print("=" * 70)
    print(f"max_depth = {max_depth}")
    print(f"Precisión en datos de prueba: {precision:.4f} ({precision*100:.2f}%)")
    print("\nReglas aprendidas:")
    print(export_text(tree, feature_names=list(wine.feature_names)))
    return precision, tree

def main():
    print("CLASIFICACIÓN CON ÁRBOL DE DECISIÓN - WINE DATASET")
    print("Dataset: 178 muestras, 13 características y 3 clases.\n")

    # Árbol solicitado en la actividad
    precision_2, tree_2 = entrenar_y_mostrar(2)

    # Comparación de diferentes profundidades
    print("\n\nCOMPARACIÓN DE max_depth")
    resultados = []
    for depth in [1, 2, 3, 4, 5, None]:
        precision, tree = entrenar_y_mostrar(depth)
        resultados.append((depth, precision, tree.get_depth(), tree.get_n_leaves()))

    print("\nRESUMEN")
    print("max_depth | precisión | profundidad_real | hojas")
    print("-" * 55)
    for depth, precision, real_depth, leaves in resultados:
        print(f"{str(depth):>9} | {precision:.4f}    | {real_depth:>16} | {leaves:>5}")

if __name__ == "__main__":
    main()
