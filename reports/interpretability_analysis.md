# Interpretability & Per-User Analysis

## Global: what do the latent factors mean?

`factor_genre_heatmap.png` shows the average PMF item-factor value per genre. `latent_factor_examples.csv` lists the top-5 movies loading most strongly on each of the first 5 factors, to sanity-check the theme a factor-genre correlation implies.

## Local: why was a specific movie recommended?

For user 1: "Gone with the Wind (1939)" was recommended mainly Because you rated "Sound of Music, The (1965)" 5/5 (similarity 0.41); and Because you rated "Gigi (1958)" 4/5 (similarity 0.38); and Because you rated "Cinderella (1950)" 5/5 (similarity 0.37)

See `why_recommended_user_1.png`.

## Three-user comparison

- **Accurate (training) user 2536**: test RMSE=0.263, 44 training ratings, rating std=0.73, top-3 genres cover 77% of their ratings (['Comedy', 'Drama', 'Romance']). Consistent, concentrated taste is easy for a low-rank model to capture.
- **Inaccurate (training) user 2033**: test RMSE=1.961, 45 training ratings, rating std=1.37, top-3 genres cover only 60% of their ratings (['Comedy', 'Drama', 'Sci-Fi']). Higher rating variance and more scattered genre taste are harder for the same latent factors to explain, hence the higher error.
- **Cold-start (simulated test-only) user 703**: RMSE=0.808 over 20 ratings, from a model retrained with every one of this user's ratings excluded from training. Their bias term stayed at exactly 0 and their latent vector stayed at random-init scale (norm=0.61, never touched by a gradient update) -- direct proof no personalized signal was learned. Their full prediction vector correlates 0.984 with the pure global-mean + item-bias baseline, versus 0.788 for real trained user 1 -- predictions for a cold-start user track generic item popularity noticeably more closely than a real trained user's do. RMSE alone can look deceptively reasonable for a cold-start user (rating scale and item biases already explain a lot of variance), which is exactly why this correlation check, not just RMSE, is the meaningful evidence of the cold-start limitation.
