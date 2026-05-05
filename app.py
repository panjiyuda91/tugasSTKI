# Tambahkan ke selectbox
method = st.selectbox(
    "Pilih metode retrieval:",
    ["TF-IDF", "BM25", "Semantic (Dense)", "Hybrid (RRF) ⭐"]
)

if st.button("Cari") and query:
    if method == "TF-IDF":
        results = retrieve_tfidf(query, ...)

    elif method == "BM25":
        results = retrieve_bm25(query, ...)

    elif method == "Semantic (Dense)":
        results = retrieve_dense(query, ...)

    else:  # Hybrid RRF
        tfidf_r = retrieve_tfidf(query, ...)
        bm25_r  = retrieve_bm25(query, ...)
        dense_r = retrieve_dense(query, ...)

        rrf_ranking = reciprocal_rank_fusion(
            build_ranked_id_list(tfidf_r),
            build_ranked_id_list(bm25_r),
            build_ranked_id_list(dense_r),
        )
        # Ambil DataFrame dari ID hasil RRF
        rrf_ids = [doc_id for doc_id, _ in rrf_ranking]
        results = df[df['Place_Id'].isin(rrf_ids)].copy()
        results['score'] = results['Place_Id'].map(dict(rrf_ranking))
        results = results.sort_values('score', ascending=False)

    # Render hasil (sama untuk semua metode)
    for _, row in results.iterrows():
        with st.expander(f"📍 {row['Place_Name']} — {row['City']}"):
            st.metric("Skor Relevansi", f"{row['score']:.4f}")
            st.write(row['Description'])