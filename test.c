
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_SAMPLES 6
#define N_FEATURES 3

/*Modelo*/

typedef struct {
    int n_features;
    double *w;
    double b;
} LogisticModel;

/* Prototypes to ensure const-correct signatures are visible to the compiler */
double sigmoid(double z);
double predict_proba(const LogisticModel *model, const double *x);
int predict_label(const LogisticModel *model, const double *x, double threshold);


/* Datos de ejemplo (subconjunto reducido) */
// X: 6 muestras x 3 características
//double X[N_SAMPLES][N_FEATURES] = {
//    {0.5,  1.2, -0.3},
//   {1.0, -0.8,  0.7},
//    {-0.5, 0.3,  1.5},
//    {-1.2, 0.4, -0.7},
//    {0.3,  0.9,  1.0},
//    {-0.7,-1.1, -0.4}
//};

// y: etiquetas binarias
//int y[N_SAMPLES] = {1, 1, 1, 0, 0, 0};

/*Dataset*/
typedef struct {
    int n_samples;
    int n_features;
    double **X;  // X[i][j]
    int *y;      // y[i]
} Dataset;
/*Funcion para crear un arreglo en memoria para alojar el dataset*/
Dataset allocate_dataset(int n_samples, int n_features) {
    Dataset dataset;
    dataset.n_samples = n_samples;
    dataset.n_features = n_features;

    dataset.X = (double **)malloc(n_samples * sizeof(double *));
    for (int i = 0; i < n_samples; i++) {
        dataset.X[i] = (double *)malloc(n_features * sizeof(double));
    }

    dataset.y = (int *)malloc(n_samples * sizeof(int));

    return dataset;
}
// Funcion para liberar memoria del dataset
void free_dataset(Dataset *dataset) {
    for (int i = 0; i < dataset->n_samples; i++) {
        free(dataset->X[i]);
    }
    free(dataset->X);
    free(dataset->y);
    dataset->X = NULL;
    dataset->y = NULL;
}


// Cargar dataset desde archivo CSV
Dataset load_dataset(const char *filename, int n_samples, int n_features) {
    Dataset data = allocate_dataset(n_samples, n_features);

    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("Error abriendo archivo %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < n_samples; ++i) {

        for (int j = 0; j < n_features; ++j) {
            if (fscanf(f, "%lf,", &data.X[i][j]) != 1) {
                printf("Error leyendo X en fila %d, col %d\n", i, j);
                exit(1);
            }
        }
        // La etiqueta va al final 
        if (fscanf(f, "%d", &data.y[i]) != 1) {
            printf("Error leyendo y en fila %d\n", i);
            exit(1);
        }
    }

    fclose(f);
    return data;
}

void init_model(LogisticModel *model, int n_features) {
    model->n_features = n_features;
    model->w = (double *)malloc(n_features * sizeof(double));
    model->b = 0.0;


    for (int i = 0; i < n_features; i++) {
        model->w[i] = 0.0;
    }
}

/* Función sigmoide */
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}


/* Predicción */

double predict_proba(const LogisticModel *model, const double *x) {
    double z = model->b;
    for (int i = 0; i < model->n_features; i++) {
        z += model->w[i] * x[i];
    }
    return sigmoid(z);
}
// Predicción de etiqueta basada en umbral
int predict_label(const LogisticModel *model, const double *x, double threshold) {
    double p = predict_proba(model, x);
    return (p >= threshold) ? 1 : 0;
}

void train_logreg(LogisticModel *model,
                  const Dataset *data,
                  double learning_rate,
                  int epochs) {

    int n = data->n_samples;
    int d = data->n_features;

    double *grad_w = (double *)malloc(d * sizeof(double));

    for (int epoch = 0; epoch < epochs; ++epoch) {

        // Inicializar gradientes en 0
        for (int j = 0; j < d; ++j) grad_w[j] = 0.0;
        double grad_b = 0.0;

        // Recorremos todas las muestras
        for (int i = 0; i < n; ++i) {
            double *x = data->X[i];
            int y = data->y[i];

            double z = model->b;
            for (int j = 0; j < d; ++j) {
                z += model->w[j] * x[j];
            }
            double p = sigmoid(z);
            double error = p - (double)y; // (p - y) es el gradiente respecto a z

            // Acumulamos gradientes
            for (int j = 0; j < d; ++j) {
                grad_w[j] += error * x[j];
            }
            grad_b += error;
        }

        // Promediamos gradientes
        for (int j = 0; j < d; ++j) {
            grad_w[j] /= (double)n;
        }
        grad_b /= (double)n;

        // Actualizamos pesos y sesgo
        for (int j = 0; j < d; ++j) {
            model->w[j] -= learning_rate * grad_w[j];
        }
        model->b -= learning_rate * grad_b;
    }

    free(grad_w);
}

// accuracy = correct_predictions / total_predictions
double accuracy_dataset(const LogisticModel *model, const Dataset *data) {
    int correct = 0;
    for (int i = 0; i < data->n_samples; ++i) {
        int y_hat = predict_label(model, data->X[i], 0.5);
        if (y_hat == data->y[i]) correct++;
    }
    return (double)correct / (double)data->n_samples;
}

void confusion_matrix(const LogisticModel *model,
                      const Dataset *data,
                      int *tn, int *fp, int *fn, int *tp) {

    *tn = *fp = *fn = *tp = 0;

    for (int i = 0; i < data->n_samples; ++i) {
        int y_true = data->y[i];
        int y_pred = predict_label(model, data->X[i], 0.5);

        if (y_true == 0 && y_pred == 0) (*tn)++;
        else if (y_true == 0 && y_pred == 1) (*fp)++;
        else if (y_true == 1 && y_pred == 0) (*fn)++;
        else if (y_true == 1 && y_pred == 1) (*tp)++;
    }
}


int main() {

    int n_samples = 3577;
    int n_features = 3;
    int n_test = 1533;
    
    Dataset train = load_dataset("data_ministroke.csv", n_samples, n_features);
    Dataset test = load_dataset("data_ministroke-test.csv", n_test, n_features);    

    LogisticModel model;
    init_model(&model, N_FEATURES);

    double learning_rate = 0.1;
    int epochs = 1000;

    //entrenamiento
    train_logreg(&model, &train, learning_rate, epochs);

    int tn, fp, fn, tp;
    confusion_matrix(&model, &train, &tn, &fp, &fn, &tp);
    printf("Matriz de confusión (train): TN=%d, FP=%d, FN=%d, TP=%d\n", tn, fp, fn, tp);

    double acc_train = accuracy(&model, &train);
    printf("Accuracy: %.2f%%\n", acc_train * 100.0);
    
    double acc_test = accuracy(&model, &train);
    printf("Accuracy: %.2f%%\n", acc_test * 100.0);
    
    
    free(model.w);
    free_dataset(&train);
    free_dataset(&test);
    return 0;
}
