def run_shapley(folder_name, n_gram=3):
    enriched_accuracy_values = []
    modified_accuracy_values = []
    male_shapley_scores = []
    female_shapley_scores = []
    male_shapley_scores_modified = []
    female_shapley_scores_modified = []
    df = load_data(folder_name, "df.pkl")
    categorical_attributes, numerical_attributes = get_attributes(folder_name)
    critical_decisions = get_critical_decisions(folder_name)
    X_enriched, y_enriched, class_names, feature_names, feature_indices = process_df(df, categorical_attributes, numerical_attributes, n_gram=n_gram)
    if isinstance(class_names, list):
        output_index = class_names.index("A_DECLINED")
    elif isinstance(class_names, np.ndarray):
        output_index = np.where(class_names == "A_DECLINED")[0][0]


    for _ in tqdm(range(3), desc="evaluating model:"):
        # evaluating the enriched model
        X_train, X_test, y_train, y_test = train_test_split(X_enriched, y_enriched, test_size=0.3)
        nn = train_nn(X_train, y_train)
        y_distilled = distill_nn(nn, X_train)
        dt_distilled = train_dt(X_train, y_distilled, class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
        enriched_accuracy = evaluate_nn(nn, X_test, y_test)
        enriched_accuracy_values.append(enriched_accuracy)
        shapley_scores = calculate_shapley_scores(nn, X_test, feature_names, feature_names, output_index)
        male_shapley_score = shapley_scores["gender = male"]
        female_shapley_score = shapley_scores["gender = female"]
        male_shapley_scores.append(male_shapley_score)
        female_shapley_scores.append(female_shapley_score)

        # modifying the distilled model
        nodes_to_remove = dt_distilled.find_nodes_to_remove(critical_decisions)
        if nodes_to_remove:
            print(f"Removing nodes: {nodes_to_remove}")
            y_encoded = np.argmax(y_train, axis=1)
            for node_id in nodes_to_remove:
                dt_distilled.delete_node(X_train, y_encoded, node_id)
            y_modified = dt_distilled.predict(X_train)
            y_encoded = to_categorical(y_modified, num_classes=len(dt_distilled.class_names))

            # finetuning and evaluating the changed model
            nn_modified = finetune_nn(nn, X_train, y_encoded, y_distilled=y_train, mode="changed")
            modified_accuracy = evaluate_nn(nn_modified, X_test, y_test)
            shapley_scores = calculate_shapley_scores(nn_modified, X_test, feature_names, feature_names, output_index)
            male_shapley_score_modified = shapley_scores["gender = male"]
            female_shapley_score_modified = shapley_scores["gender = female"]
        else:
            print(f"No nodes to remove!")
            male_shapley_score_modified = male_shapley_score
            female_shapley_score_modified = female_shapley_score
            modified_accuracy = enriched_accuracy

        male_shapley_scores_modified.append(male_shapley_score_modified)
        female_shapley_scores_modified.append(female_shapley_score_modified)
        modified_accuracy_values.append(modified_accuracy)

        print(f"enriched accuracy: {enriched_accuracy}, modified accuracy: {modified_accuracy}")
        print(f"enriched: male shapley score: {male_shapley_score}, female shapley score: {female_shapley_score}")
        print(f"modified: male shapley score: {male_shapley_score_modified}, female shapley score: {female_shapley_score_modified}")
        print("--------------------------------------------------------------------------------------------------")

    save_data(male_shapley_scores, folder_name, "male_shapley_scores.pkl")
    save_data(female_shapley_scores, folder_name, "female_shapley_scores.pkl")
    save_data(male_shapley_scores_modified, folder_name, "male_shapley_scores_modified.pkl")
    save_data(female_shapley_scores_modified, folder_name, "female_shapley_scores_modified.pkl")
    save_data(enriched_accuracy_values, folder_name, "enriched_accuracy_values.pkl")
    save_data(modified_accuracy_values, folder_name, "modified_accuracy_values.pkl")
    plot_shapley(male_shapley_scores, male_shapley_scores_modified, folder_name, "shapley_male.png")
    plot_shapley(female_shapley_scores, female_shapley_scores_modified, folder_name, "shapley_female.png")


def run_fairness(folder_name):
    nn_base = load_nn(folder_name, "nn_base.keras")
    nn_enriched = load_nn(folder_name, "nn_enriched.keras")
    nn_modified = load_nn(folder_name, "nn_modified.keras")
    X_test  = load_data(folder_name, "X_test.pkl")
    class_names = load_data(folder_name, "class_names.pkl")
    feature_names = load_data(folder_name, "feature_names.pkl")
    feature_indices = load_data(folder_name, "feature_indices.pkl")
    critical_decisions = get_critical_decisions(folder_name)
    base_attributes = ["case:AMOUNT_REQ"]
    disp_imp_results = {}

    disp_imp_result = calculate_comparable_fairness(nn_base, nn_enriched, nn_modified, X_test, critical_decisions, feature_indices, class_names, feature_names, base_attributes)
    print(disp_imp_result)
    for outer_key, outer_value in disp_imp_result.items():
        disp_imp_results[outer_key] = {}
        for inner_key, inner_value in outer_value.items():
            disp_imp_results[outer_key][inner_key] = []
            disp_imp_results[outer_key][inner_key].append(inner_value)
    print(disp_imp_results)
    #plot_all_fairness(stat_par_results, disp_imp_results, folder_name)


def run_transformer(folder_name):
    #df = load_xes_to_df("hospital_billing")
    #process_model = build_process_model(folder_name)
    #trace_generator = TraceGenerator(process_model=process_model)
    #generated_cases = trace_generator.generate_traces(start_time=datetime.now(), num_cases=num_cases)
    #df = cases_to_dataframe(generated_cases)
    #df = load_data(folder_name, "df.pkl")
    #print(df.head(20))
    df = load_data(folder_name, "df.pkl")
    print(df.head(20))
    padding_value = len(set(df['activity']))
    vocab_size = padding_value + 1
    categorical_attributes = ["gender"]
    numerical_attributes = ["age"]
    # Split the DataFrame by case_id to ensure no case is split between train and test
    unique_case_ids = df['case_id'].unique()
    train_case_ids, test_case_ids = train_test_split(unique_case_ids, test_size=0.2, random_state=42)

    # Create train and test DataFrames
    train_df = df[df['case_id'].isin(train_case_ids)]
    test_df = df[df['case_id'].isin(test_case_ids)]

    # Process the training data
    input_sequences_train, target_sequences_train, categorical_data_train, numerical_data_train = process_data_padded(
        train_df, categorical_attributes, numerical_attributes
    )
    max_seq_len = input_sequences_train.shape[1]
    print(f"Max length: {max_seq_len}")

    print("First 20 input sequences:")
    print(input_sequences_train[:20])

    print("\nFirst 20 target sequences:")
    print(target_sequences_train[:20])

    print("\nFirst 20 categorical data entries:")
    for i, cat_data in enumerate(categorical_data_train[:20]):
        print(f"Categorical attribute {i}: {cat_data}")

    print("\nFirst 20 numerical data entries:")
    for i, num_data in enumerate(numerical_data_train[:20]):
        print(f"Numerical attribute {i}: {num_data}")

    # Process the test data
    input_sequences_test, target_sequences_test, categorical_data_test, numerical_data_test = process_data_padded(
        test_df, categorical_attributes, numerical_attributes, max_seq_len=max_seq_len
    )

    # Create the transformer model
    model = build_transformer_model(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        padding_value=padding_value,
        num_categorical=len(categorical_attributes),
        num_numerical=len(numerical_attributes),
        embed_dim=32,
        num_heads=2,
        ff_dim=64
    )

    # Train the model
    history = model.fit(
        [input_sequences_train] + categorical_data_train + numerical_data_train,
        target_sequences_train,
        batch_size=32,
        epochs=10,
        validation_split=0.2
    )

    # Evaluate the model
    results = model.evaluate(
        [input_sequences_test] + categorical_data_test + numerical_data_test,
        target_sequences_test
    )

    # Calculate and print accuracy
    print(f"Test Accuracy: {results[1] * 100:.2f}%")
    

def run_sklearn_test(folder_name="cc", n_gram=2, num_cases=10, preprocessing=False):
    if preprocessing:
        run_preprocessing(folder_name, n_gram=n_gram, num_cases=num_cases, console_output=False)
    nn = load_nn(folder_name, "nn.keras")
    X_train  = load_data(folder_name, "X_train.pkl")
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    class_names = load_data(folder_name, "class_names.pkl")
    feature_names = load_data(folder_name, "feature_names.pkl")
    feature_indices = load_data(folder_name, "feature_indices.pkl")

    y_distilled = distill_nn(nn, X_train)
    start_time = time.time()
    dt_sklearn = train_sklearn_dt(X_train, y_distilled)
    end_time = time.time()
    print(f"Training time for sklearn decision tree: {end_time - start_time:.2f} seconds")
    start_time = time.time()
    dt_distilled = train_dt(X_train, y_distilled, folder_name=folder_name, model_name="dt_distilled.json", class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
    end_time = time.time()
    print(f"Training time for distilled decision tree: {end_time - start_time:.2f} seconds")
    start_time = time.time()
    dt_transformed = sklearn_to_custom_tree(dt_sklearn, feature_names=feature_names, class_names=class_names, feature_indices=feature_indices)
    end_time = time.time()
    print(f"Training time for transformed decision tree: {end_time - start_time:.2f} seconds")

    print("Base NN:")
    evaluate_nn(nn, X_test, y_test)
    print("Directly distilled DT:")
    evaluate_dt(dt_distilled, X_test, y_test)
    print("Sklearn distilled DT:")
    evaluate_sklearn_dt(dt_sklearn, X_test, y_test, y_distilled, class_names=class_names, feature_names=feature_names)

    print("Transformed distilled DT:")
    evaluate_dt(dt_transformed, X_test, y_test)



def evaluate_sklearn_dt(dt, X_test, y_test, y_distilled, class_names=None, feature_names=None):
    print("testing dt:")
    y_argmax = np.argmax(y_test, axis=1)
    accuracy = dt.score(X_test, y_argmax)
    seen = set()
    sklearn_class_names = [
        class_names[i] for i in y_distilled if i < len(class_names) and not (i in seen or seen.add(i))
    ]
    sklearn_class_names.sort()
    print(f"Test set accuracy: {accuracy:.2f}")
    tree_text = export_text(dt, class_names=sklearn_class_names, feature_names=feature_names)
    print(tree_text)

def masked_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    loss *= mask  # Only consider non-padding tokens
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    correct_predictions = tf.cast(
        tf.equal(tf.argmax(y_pred, axis=-1), tf.cast(y_true, tf.int64)), dtype=tf.float32
    )
    correct_predictions *= mask  # Only consider non-padding tokens
    return tf.reduce_sum(correct_predictions) / tf.reduce_sum(mask)


def calculate_shapley_scores(model, X, relevant_attributes, feature_names, output_index, background_size=1000):
    # Verify that relevant attributes are in feature_names
    assert all(attr in feature_names for attr in relevant_attributes), "Relevant attributes must exist in feature names."

    # Map relevant attributes to their indices in X
    relevant_indices = [feature_names.index(attr) for attr in relevant_attributes]
    print(relevant_indices)

    if X.shape[0] > background_size:
        X = X[np.random.choice(X.shape[0], size=background_size, replace=False), :]

    # Initialize SHAP DeepExplainer
    explainer = shap.DeepExplainer(model, X)

    # Compute SHAP values
    shap_values = explainer.shap_values(X)  # Returns a list of arrays (one for each output class)
    print(shap_values.shape)

    # For simplicity, aggregate shap values across all classes (sum of absolute values)
    #shap_values = np.sum(np.abs(np.array(shap_values)), axis=0)
    shap_values = shap_values[:, :, output_index]
    print(shap_values.shape)

    shapley_scores = {}
    for i, idx in enumerate(relevant_indices):
        shapley_scores[relevant_attributes[i]] = np.mean(np.abs(shap_values[:, idx]))
        if i in [34, 35]:
            plot_density(shap_values[:, idx], idx)
    print(shapley_scores)

    return shapley_scores


def build_transformer_model(vocab_size, max_seq_len, padding_value, num_categorical=0, num_numerical=0, embed_dim=32, num_heads=2, ff_dim=64, dropout_rate=0.1):
    """
    Creates a transformer model for next activity prediction.

    Args:
        vocab_size (int): Number of unique activities.
        max_seq_len (int): Maximum sequence length.
        num_categorical (int): Number of categorical features.
        num_numerical (int): Number of numerical features.
        embed_dim (int): Embedding dimension for activities.
        num_heads (int): Number of attention heads.
        ff_dim (int): Feed-forward network dimension.
        dropout_rate (float): Dropout rate.

    Returns:
        tf.keras.Model: Compiled transformer model.
    """
    # Embedding layer with mask_zero=True creates a mask
    activity_input = Input(shape=(max_seq_len,), name="activity_input")
    activity_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(activity_input)

    # Extract the mask from the embedding layer
    mask = activity_embedding._keras_mask  # Keras propagates this mask automatically

    # Positional Encoding
    position_input = tf.range(start=0, limit=max_seq_len, delta=1)
    position_embedding = Embedding(input_dim=max_seq_len, output_dim=embed_dim)(position_input)
    x = activity_embedding + position_embedding

    # Transformer Encoder Block
    mask = ExpandDimsLayer(axis=1)(mask)  # Adjust dimensions for MultiHeadAttention

    # Transformer Encoder Block
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        activity_embedding, activity_embedding, attention_mask=mask
    )
    attn_output = Dropout(dropout_rate)(attn_output)
    attn_output = LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed-forward network
    ff_output = Dense(ff_dim, activation='relu')(attn_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(embed_dim)(ff_output)
    x = LayerNormalization(epsilon=1e-6)(attn_output + ff_output)

    # Concatenate additional features
    categorical_inputs, numerical_inputs = [], []
    if num_categorical > 0:
        # Create inputs and embeddings for categorical features
        cat_inputs = [Input(shape=(max_seq_len,), name=f"cat_input_{i}") for i in range(num_categorical)]
        cat_embeddings = [Embedding(100, embed_dim)(cat_input) for cat_input in cat_inputs]
        x = Concatenate(axis=-1)([x] + cat_embeddings)
        categorical_inputs.extend(cat_inputs)

    # Handle numerical inputs
    if num_numerical > 0:
        # Create inputs with shape (max_seq_len, 1) to include time dimension
        num_inputs = [Input(shape=(max_seq_len, 1), name=f"num_input_{i}") for i in range(num_numerical)]
        num_dense = [Dense(embed_dim)(num_input) for num_input in num_inputs]
        x = Concatenate(axis=-1)([x] + num_dense)
        numerical_inputs.extend(num_inputs)

    # Output layer
    output = Dense(vocab_size, activation='softmax', name='output')(x)

    # Define model
    inputs = [activity_input] + categorical_inputs + numerical_inputs
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss=masked_loss, metrics=[masked_accuracy])

    return model


def run_interactive():
    print("Python Interactive Shell. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            # Read user input
            command = input(">>> ")

            # Exit the shell if the user types 'exit' or 'quit'
            if command in ["exit", "quit"]:
                print("Exiting the shell.")
                break

            # Attempt to evaluate expressions (for things like math, variables, etc.)
            try:
                result = eval(command)
                if result is not None:
                    print(result)
            except SyntaxError:
                # If eval fails due to syntax error, it's likely a statement (e.g., assignment, import)
                exec(command)
        
        # Catch and display any runtime errors
        except Exception as e:
            print(f"Error: {e}")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting the shell.")
            break

def plot_shapley(enriched_accuracy_values, modified_accuracy_values, folder_name, file_name):
    img_folder = os.path.join("img", folder_name)
    os.makedirs(img_folder, exist_ok=True)
    
    data = {
        'Enriched': np.array(enriched_accuracy_values),
        'Modified': np.array(modified_accuracy_values)
    }
    
    # Calculate mean and standard deviation
    stats = {name: (np.mean(values), np.std(values)) for name, values in data.items()}
    print("Accuracy Statistics:")
    for name, (mean, std) in stats.items():
        print(f"{name}: Mean = {mean:.3f}, Std Dev = {std:.3f}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Define custom colors
    for i, (name, values) in enumerate(data.items()):
        sns.kdeplot(values, label=f'{name} (μ={stats[name][0]:.3f}, σ={stats[name][1]:.3f})',
                    color=colors[i], fill=True, alpha=0.6, linewidth=2)
    
    # Add titles and labels
    plt.title(f'Shapley Score of {folder_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Shapley', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12, title='Legend', title_fontsize='13')
    
    # Save the plot
    plt.tight_layout()
    image_path = os.path.join('img', folder_name, file_name)
    plt.savefig(image_path)
    plt.close()

def plot_accuracy(base_accuracy_values, enriched_accuracy_values, modified_accuracy_values, folder_name):
    img_folder = os.path.join("img", folder_name)
    os.makedirs(img_folder, exist_ok=True)
    
    data = {
        'Base': np.array(base_accuracy_values),
        'Enriched': np.array(enriched_accuracy_values),
        'Modified': np.array(modified_accuracy_values)
    }
    
    # Calculate mean and standard deviation
    stats = {name: (np.mean(values), np.std(values)) for name, values in data.items()}
    print("Accuracy Statistics:")
    for name, (mean, std) in stats.items():
        print(f"{name}: Mean = {mean:.3f}, Std Dev = {std:.3f}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Define custom colors
    for i, (name, values) in enumerate(data.items()):
        sns.kdeplot(values, label=f'{name} (μ={stats[name][0]:.3f}, σ={stats[name][1]:.3f})',
                    color=colors[i], fill=True, alpha=0.6, linewidth=2)
    
    # Add titles and labels
    plt.title(f'Accuracy Distribution of {folder_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Accuracy', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12, title='Legend', title_fontsize='13')
    
    # Save the plot
    plt.tight_layout()
    image_path = os.path.join('img', folder_name, f"accuracy.png")
    plt.savefig(image_path)
    plt.close()

def k_fold_evaluation(df, critical_decisions, categorical_attributes, numerical_attributes, base_attributes, folds=5, n_gram=3, modify_mode="retrain", finetuning_mode="changed_complete", folder_name=None):
    base_accuracy_values = []
    enriched_accuracy_values = []
    modified_accuracy_values = []
    node_counts = []
    stat_par_results = {}
    class_names = sorted(df["activity"].unique().tolist() + ["<PAD>"])
    attribute_pools = create_attribute_pools(df, categorical_attributes)
    feature_names = create_feature_names(class_names, attribute_pools, numerical_attributes, n_gram)
    feature_indices = create_feature_indices(class_names, attribute_pools, numerical_attributes, n_gram)
    print(class_names)
    print(feature_names)
    print(feature_indices)

    folds = k_fold_cross_validation(df, categorical_attributes, numerical_attributes, critical_decisions, n_gram=3, k=folds)

    for i, (X_train, y_train, X_test, y_test, numerical_thresholds) in tqdm(enumerate(folds), desc="evaluating model:"):
        # evaluating the base model
        X_train_base = remove_attribute_features(X_train, feature_indices, base_attributes)
        X_test_base = remove_attribute_features(X_test, feature_indices, base_attributes)
        nn_base = train_nn(X_train_base, y_train)
        base_accuracy = evaluate_nn(nn_base, X_test_base, y_test)
        base_accuracy_values.append(base_accuracy)

        # evaluating the enriched model
        nn_enriched = train_nn(X_train, y_train)
        y_distilled = nn_enriched.predict(X_train)
        y_encoded = np.argmax(y_distilled, axis=1)

        dt_distilled = train_dt(X_train, y_encoded, class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
        enriched_accuracy = evaluate_nn(nn_enriched, X_test, y_test)
        enriched_accuracy_values.append(enriched_accuracy)

        evaluate_dt(dt_distilled, X_test, y_test)
        num_nodes = dt_distilled.count_nodes()
        node_counts.append(num_nodes)

        # modifying the distilled model
        nodes_to_remove = dt_distilled.find_nodes_to_remove(critical_decisions)
        y_distilled_tree = dt_distilled.predict(X_train)
        y_distilled_tree = to_categorical(y_distilled_tree, num_classes=len(dt_distilled.class_names))

        # prepare modified model
        nn_modified = clone_model(nn_enriched)
        nn_modified.set_weights(nn_enriched.get_weights())
        nn_modified_cut = clone_model(nn_enriched)
        nn_modified_cut.set_weights(nn_enriched.get_weights())
        nn_modified_retrain = clone_model(nn_enriched)
        nn_modified_retrain.set_weights(nn_enriched.get_weights())
        if nodes_to_remove:
            print(f"Removing nodes: {nodes_to_remove}")
            dt_distilled_retrain = dt_distilled
            dt_distilled_cut = copy_decision_tree(dt_distilled)
            y_encoded = np.argmax(y_train, axis=1)
            for node_id in nodes_to_remove:
                dt_distilled_retrain.delete_node(X_train, y_encoded, node_id)
                #evaluate_dt(dt_distilled_retrain, X_test, y_test)
                dt_distilled_cut.delete_branch(node_id)
            evaluate_dt(dt_distilled_retrain, X_test, y_test)
            y_modified_cut = dt_distilled_cut.predict(X_train)
            y_modified_cut = to_categorical(y_modified_cut, num_classes=len(dt_distilled.class_names))
            y_modified_retrain = dt_distilled_retrain.predict(X_train)
            y_modified_retrain = to_categorical(y_modified_retrain, num_classes=len(dt_distilled.class_names))

            # finetuning and evaluating the best changed model
            nn_modified_cut = finetune_all(nn_modified_cut, X_train, y_modified_cut, y_distilled_tree, y_distilled, X_test, y_test)
            nn_modified_retrain = finetune_all(nn_modified_retrain, X_train, y_modified_retrain, y_distilled_tree, y_distilled, X_test, y_test)
            accuracy_cut = evaluate_nn(nn_modified_cut, X_test, y_test)
            accuracy_retrain = evaluate_nn(nn_modified_retrain, X_test, y_test)
            print(f"Accuracy Cut: {accuracy_cut}, Accuracy Retrain: {accuracy_retrain}")
            if accuracy_cut > accuracy_retrain:
                print("Picking cut tree!")
                nn_modified = nn_modified_cut
                modified_accuracy = accuracy_cut
            else:
                print("Picking retrained tree!")
                nn_modified = nn_modified_retrain
                modified_accuracy = accuracy_retrain
        else:
            print(f"No nodes to remove!")
            modified_accuracy = enriched_accuracy
        modified_accuracy_values.append(modified_accuracy)

        print(f"base accuracy: {base_accuracy}, enriched accuracy: {enriched_accuracy}, modified accuracy: {modified_accuracy}")
        calculate_fairness(nn_modified, X_test, critical_decisions, feature_indices, class_names, feature_names)
        stat_par_result, _ = calculate_comparable_fairness(nn_base, nn_enriched, nn_modified, X_test, critical_decisions, feature_indices, class_names, feature_names, base_attributes, numerical_thresholds)
        for outer_key, outer_value in stat_par_result.items():
            if i == 0:
                stat_par_results[outer_key] = {}
            for inner_key, inner_value in outer_value.items():
                if i == 0:
                    stat_par_results[outer_key][inner_key] = []
                stat_par_results[outer_key][inner_key].append(inner_value)
        print("--------------------------------------------------------------------------------------------------")

    node_counts_array = np.array(node_counts)
    average_nodes = np.mean(node_counts_array)
    std_dev_nodes = np.std(node_counts_array)
    print(f"Average number of nodes: {average_nodes:.2f}")
    print(f"Standard deviation of nodes: {std_dev_nodes:.2f}")
    if folder_name:
        save_data(node_counts_array, folder_name, "node_counts.pkl")
        save_data(base_accuracy_values, folder_name, "base_accuracy_values.pkl")
        save_data(enriched_accuracy_values, folder_name, "enriched_accuracy_values.pkl")
        save_data(modified_accuracy_values, folder_name, "modified_accuracy_values.pkl")
        save_data(stat_par_results, folder_name, "stat_par_results.pkl")
        title = f"Accuracy Distribution of {folder_name}"
        plot_distribution(base_accuracy_values, enriched_accuracy_values, modified_accuracy_values, folder_name, title, "Accuracy")
        plot_all_parity(stat_par_results, folder_name)
    return base_accuracy_values, enriched_accuracy_values, modified_accuracy_values, stat_par_results
