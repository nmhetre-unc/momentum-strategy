"""
Per-page orientation content, as plain data.

Every page opens with the same four things -- a breadcrumb, a short "why
this matters", a pointer for anyone lost, and a list of common mistakes --
and closes with the same "what to do next" grid. Holding that content in
one registry is what makes the pages consistent by construction rather
than by discipline: there is no way for one page to drift into a different
voice or structure, because none of them own the copy.

No Streamlit import here, matching quant_notes.py. The rendering helpers
live in regime_dashboard.py.

Each entry provides:
    title / icon / path   identity, mirrored from app.py's navigation
    section               sidebar group, for the breadcrumb
    step                  position in the 6-step path, or None for reference pages
    teaches / why / habit  the three sentences of "Why this matters"
    confused              note key, glossary pointer and exercise for the lost
    mistakes              2-3 beginner mistakes specific to this page
    next                  ordered page keys for the closing grid
    next_blurbs           one line per destination, written from THIS page
"""

# The six-step sequence a learner works through. Start here, Exercises and
# Learn sit outside it: one is the front door, two are used throughout.
PAGE_ORDER = ["backtest", "regimes", "adaptive", "ml_lab", "validation", "exercises"]

PAGE_GUIDE = {
    # ----------------------------------------------------------------------
    "start_here": {
        "title": "Start here",
        "icon": ":material/rocket_launch:",
        "path": "app_pages/start_here.py",
        "section": None,
        "step": None,
        "teaches": (
            "How this platform is organised, what each page is for, and the order to work "
            "through them in."
        ),
        "why": (
            "The pages build on each other. Reading a per-regime table means little until you "
            "can read an ordinary equity curve, and adapting a strategy means little until you "
            "know which regime it struggles in."
        ),
        "habit": "Orienting before acting — knowing what question a tool answers before you run it.",
        "confused": {
            "note": "equity_curve",
            "glossary": "Learn → Glossary defines every metric you will meet.",
            "exercise": None,
        },
        "mistakes": [
            ("Skipping to the interesting pages",
             "Regimes and ML look more exciting than Backtest, and both assume you can already "
             "read an equity curve, a drawdown chart and an exposure figure. Start at step 1."),
            ("Treating the checklist as the curriculum",
             "The checklist gets you around the building in one session. The eight-stage "
             "learning path on the Learn page is the actual course."),
            ("Expecting to find a strategy that works",
             "Most attempts here lose to buy-and-hold. Recognising that quickly and saying so "
             "plainly is the skill being trained, not a failure to avoid."),
        ],
        "next": ["backtest", "learn", "exercises", "validation"],
        "next_blurbs": {
            "backtest": (
                "Start with `sma_crossover` on SPY. Read the equity curve, then the drawdown "
                "chart under it, then the buy-and-hold comparison."
            ),
            "learn": (
                "Ten minutes on Common pitfalls now saves you from the three mistakes that "
                "produce confidently wrong results."
            ),
            "exercises": (
                "After each page, do the matching exercise while the concept is still "
                "uncomfortable. That is when practice sticks."
            ),
            "validation": (
                "Run this on any result you like the look of — especially the ones you like the "
                "look of."
            ),
        },
    },
    # ----------------------------------------------------------------------
    "backtest": {
        "title": "Backtest",
        "icon": ":material/query_stats:",
        "path": "app_pages/backtest_lab.py",
        "section": "Research",
        "step": 1,
        "teaches": (
            "How to read a strategy's result: the equity curve, the drawdown underneath it, "
            "exposure, turnover, and the comparison against simply holding the asset."
        ),
        "why": (
            "Every other page produces numbers in this format. If you cannot read them here, "
            "nothing downstream will mean anything."
        ),
        "habit": "Reading the shape of a result before its endpoint, and always against a benchmark.",
        "confused": {
            "note": "equity_curve",
            "glossary": "Learn → Glossary → Metrics explains every number on this page.",
            "exercise": "Does it actually beat buy-and-hold?",
        },
        "mistakes": [
            ("Reading the final number first",
             "The endpoint is the least informative part of an equity curve. Two strategies "
             "ending at 2.0x can have completely different characters — one steady, one flat "
             "for four years then doubling. Read the shape and the flat stretches first."),
            ("Ignoring the benchmark",
             "A strategy that made 80% while buy-and-hold made 120% did not make money. It lost "
             "40% of what doing nothing would have paid."),
            ("Quoting a Sharpe ratio built on a handful of trades",
             "A ten-year backtest containing three trades gives you three independent bets. "
             "Check the trade count before believing any risk-adjusted number."),
        ],
        "next": ["regimes", "adaptive", "ml_lab", "validation", "exercises", "learn"],
        "next_blurbs": {
            "regimes": (
                "See which market environments helped or hurt this strategy, and whether those "
                "environments are real or just noise you would be conditioning on by mistake."
            ),
            "adaptive": (
                "Test whether filtering out the bad environment, or scaling position size by "
                "volatility, improves this strategy — and whether it survives its extra turnover."
            ),
            "ml_lab": (
                "Compare the machine-learning signal against these rule-based results, and watch "
                "the random forest's training accuracy collapse out-of-sample."
            ),
            "validation": (
                "Go beyond this single split: rolling walk-forward gives ten or more consecutive "
                "holdouts, plus a like-for-like comparison of all eleven strategies."
            ),
            "exercises": (
                "Ten guided exercises with automated checks, run against exactly the data you "
                "have loaded — including two that use this page directly."
            ),
            "learn": (
                "The full learning path, the three pitfalls, every quant note in one place, and "
                "a glossary of every metric on this page."
            ),
        },
    },
    # ----------------------------------------------------------------------
    "regimes": {
        "title": "Regimes",
        "icon": ":material/layers:",
        "path": "app_pages/regimes.py",
        "section": "Research",
        "step": 2,
        "teaches": (
            "How to detect persistent market environments, how to check whether they are real, "
            "and how to see which one a strategy's return actually came from."
        ),
        "why": (
            "A single full-period Sharpe ratio averages the conditions where a strategy thrived "
            "with the ones where it bled, and describes neither."
        ),
        "habit": "Asking *where* a return came from, rather than only how large it was.",
        "confused": {
            "note": "regimes",
            "glossary": "Learn → Glossary → Regime features explains every input the model uses.",
            "exercise": "Which regime does your strategy perform best in?",
        },
        "mistakes": [
            ("Trusting labels that flicker",
             "If episodes average a handful of days, the model found noise rather than "
                "environments. Check the average duration and the persistence diagonal before "
                "reading anything into a per-regime table."),
            ("Fitting the regime model on all of history, then backtesting on it",
             "The cluster centres then encode the future, and the backtest measures hindsight "
             "rather than skill. Keep the fit fraction below 1.0, or use walk-forward detection."),
            ("Reading a per-regime Sharpe without its day count",
             "The error bar on an annualised Sharpe is roughly sqrt(252/days). Over 100 days "
             "that is about ±1.6, wide enough to contain almost any conclusion."),
        ],
        "next": ["adaptive", "ml_lab", "validation", "backtest", "exercises", "learn"],
        "next_blurbs": {
            "adaptive": (
                "See how strategies behave differently across these environments, and whether "
                "sitting out the worst one survives its own turnover."
            ),
            "ml_lab": (
                "See whether the classifier has a real edge in one environment and none in "
                "another, or the same coin-flip accuracy everywhere."
            ),
            "validation": (
                "Watch the environment mix shift between in-sample and out-of-sample — usually "
                "the fastest way to tell a broken strategy from a changed market."
            ),
            "backtest": (
                "Return with a hypothesis and test it. The regime attribution toggle splits any "
                "strategy's profit and loss using exactly the model configured here."
            ),
            "exercises": (
                "Practise regime interpretation, and measure the lookahead gap on your own data "
                "as a number you can quote."
            ),
            "learn": (
                "Every quant note in one browser, the three pitfalls in detail, and a glossary "
                "of all fifteen regime features."
            ),
        },
    },
    # ----------------------------------------------------------------------
    "adaptive": {
        "title": "Adaptive",
        "icon": ":material/tune:",
        "path": "app_pages/adaptive_lab.py",
        "section": "Research",
        "step": 3,
        "teaches": (
            "Four ways to change a strategy's behaviour by environment — filtering, switching, "
            "re-parameterising and position sizing — and how each of them fails."
        ),
        "why": (
            "Knowing a strategy struggles in one environment raises an obvious question: can you "
            "simply not trade it there? Sometimes. The work is in checking honestly."
        ),
        "habit": "Asking whether added complexity paid for the turnover it introduced.",
        "confused": {
            "note": "adaptive",
            "glossary": "Learn → Glossary → Adaptive mechanisms summarises all seven wrappers.",
            "exercise": "Does position sizing beat signal engineering?",
        },
        "mistakes": [
            ("Crediting a smaller drawdown to skill",
             "Reducing exposure shrinks drawdown automatically — that is arithmetic. If the "
             "Sharpe ratio improved, the strategy avoided the market at the right moments. If "
             "only drawdown improved, it was simply invested less."),
            ("Comparing adaptive strategies at zero cost",
             "Adaptation buys its improved risk profile with extra trading. A frictionless "
             "comparison systematically flatters the mechanisms that trade most."),
            ("Trusting an automatic choice without reading its evidence",
             "If the winning candidate beat the runner-up by 0.05 of Sharpe over 80 days, that "
             "is a coin flip wearing a decision's clothes."),
        ],
        "next": ["validation", "regimes", "ml_lab", "backtest", "exercises", "learn"],
        "next_blurbs": {
            "validation": (
                "Check whether the adaptation survives out-of-sample across many rolling "
                "windows — and whether its decay is smaller than the plain strategy's."
            ),
            "regimes": (
                "Understand the environments behind the adaptation, and confirm they are real "
                "before trusting any rule conditioned on them."
            ),
            "ml_lab": (
                "See how machine-learning signals behave when conditioned on environment, and "
                "why splitting training data that way usually costs more than it buys."
            ),
            "backtest": (
                "Compare the adapted result against the plain rule on its own terms, with the "
                "full caveat set and guided interpretation."
            ),
            "exercises": (
                "Practise adaptive interpretation, including what happens to the ranking when "
                "transaction costs go on."
            ),
            "learn": (
                "Every quant note in one browser, including the six that cover the four "
                "adaptive mechanisms."
            ),
        },
    },
    # ----------------------------------------------------------------------
    "ml_lab": {
        "title": "ML lab",
        "icon": ":material/network_intelligence:",
        "path": "app_pages/ml_lab.py",
        "section": "Research",
        "step": 4,
        "teaches": (
            "How a classifier behaves on a problem with almost no signal: base rates, the "
            "train/test gap, and why prediction accuracy is not profit."
        ),
        "why": (
            "Machine learning is where overfitting is most visible and most instructive. "
            "Watching a model score 86% on training data and 45% on new data teaches more about "
            "model capacity than any amount of reading."
        ),
        "habit": "Comparing every accuracy figure against the base rate before believing it.",
        "confused": {
            "note": "ml_base_rate",
            "glossary": "Learn → Glossary → Metrics covers the base rate and the train/test gap.",
            "exercise": "Compare in-sample and out-of-sample ML accuracy",
        },
        "mistakes": [
            ("Reading accuracy without its base rate",
             "About 53% of days are up, so predicting \"up\" every day scores 53% with no model "
             "at all. A model at 51% has been beaten by a one-line constant."),
            ("Treating a high training accuracy as good news",
             "On a low-signal problem it is the opposite. The gap between training and test "
             "accuracy is the quantity of interest, not either level."),
            ("Assuming accuracy translates into profit",
             "A model right on 60 small days and wrong on 40 large ones is 60% accurate and "
             "loses money. Read the out-of-sample Sharpe ratio instead."),
        ],
        "next": ["validation", "adaptive", "regimes", "backtest", "exercises", "learn"],
        "next_blurbs": {
            "validation": (
                "Check whether the model survives out-of-sample across many rolling windows, and "
                "where it sits against all ten other strategies once costs are on."
            ),
            "adaptive": (
                "See how the signal behaves when sized or filtered rather than traded raw — with "
                "volatility targeting as the control group."
            ),
            "regimes": (
                "See how accuracy varies by environment, and check the environments are real "
                "before reading into any per-regime difference."
            ),
            "backtest": (
                "Compare the machine-learning signal against the simple rule-based strategies on "
                "identical data. The comparison is usually humbling."
            ),
            "exercises": (
                "Practise interpretation: compare in-sample and out-of-sample accuracy, then "
                "explain the overfitting mechanism in your own words."
            ),
            "learn": (
                "Every quant note in one browser, including the six covering base rates, "
                "bias and variance, and feature importance."
            ),
        },
    },
    # ----------------------------------------------------------------------
    "validation": {
        "title": "Validation",
        "icon": ":material/fact_check:",
        "path": "app_pages/validation.py",
        "section": "Research",
        "step": 5,
        "teaches": (
            "How to check a result survives on data that played no part in building it: the "
            "in-sample to out-of-sample gap, rolling walk-forward blocks, and fair comparison."
        ),
        "why": (
            "Everything before this point can be fooled by a lucky sample. This is the page that "
            "tells you whether you found something or fitted something."
        ),
        "habit": "Asking \"in-sample or out-of-sample?\" about every number, including your own.",
        "confused": {
            "note": "is_vs_oos",
            "glossary": "Learn → Glossary → Metrics defines Sharpe decay and exposure.",
            "exercise": "Run walk-forward validation and interpret it",
        },
        "mistakes": [
            ("Reading the levels instead of the gap",
             "A small gap at a modest level beats a large level with a large gap. The first "
             "describes a process you can repeat; the second describes one lucky period."),
            ("Treating one split as settled evidence",
             "A single holdout is one draw. It cannot distinguish a strategy that works from one "
             "tested on a friendly stretch of market. Read the rolling folds instead."),
            ("Picking the best strategy off the comparison table",
             "If you choose using the out-of-sample column, that column is no longer "
             "out-of-sample. The honest report is the whole table."),
        ],
        "next": ["regimes", "adaptive", "ml_lab", "backtest", "exercises", "learn"],
        "next_blurbs": {
            "regimes": (
                "See which environments caused the decay — and whether the strategy broke or "
                "simply met more of the environment it dislikes."
            ),
            "adaptive": (
                "Test whether filtering or position sizing improves robustness, and whether the "
                "adapted version's decay is smaller than the plain one's."
            ),
            "ml_lab": (
                "See how the machine-learning signal behaves out-of-sample, where the accuracy "
                "gap turns into a Sharpe gap that costs money."
            ),
            "backtest": (
                "Try different parameters — deciding your acceptance criterion before you look, "
                "so you are not fitting to the holdout."
            ),
            "exercises": (
                "Practise validation interpretation: commit to a reading, then check it against "
                "what the numbers actually support."
            ),
            "learn": (
                "Every quant note in one browser, including the five covering walk-forward, fold "
                "uncertainty and silent fitting."
            ),
        },
    },
    # ----------------------------------------------------------------------
    "exercises": {
        "title": "Exercises",
        "icon": ":material/assignment:",
        "path": "app_pages/exercises_lab.py",
        "section": "Training",
        "step": 6,
        "teaches": (
            "Ten guided exercises that check a specific effect against the data you have loaded, "
            "each one installing a single research habit."
        ),
        "why": (
            "Reading about an effect and measuring it yourself are different experiences. The "
            "second one is the one that changes how you work."
        ),
        "habit": "Predicting before measuring, and explaining a result in your own words.",
        "confused": {
            "note": "walkforward_reason",
            "glossary": "Learn → Glossary covers every metric these checks report.",
            "exercise": None,
        },
        "mistakes": [
            ("Running the check before forming a view",
             "Predicting first turns the check into a test of how you think markets work. "
             "Running first turns it into a demonstration you nod along with and forget."),
            ("Reading \"not confirmed\" as a failing grade",
             "Several exercises test whether a known effect appears in *your* data. Sometimes it "
             "does not, and that is a real result rather than an error."),
            ("Opening the answer before writing your own",
             "The gap between your explanation and the written one is exactly where your "
             "understanding is thin. Skipping the comparison discards the lesson."),
        ],
        "next": ["regimes", "adaptive", "ml_lab", "validation", "backtest", "learn"],
        "next_blurbs": {
            "regimes": (
                "See the environment behind your results, and check it is real before reading "
                "into any per-regime number an exercise reported."
            ),
            "adaptive": (
                "Improve a strategy with filtering or sizing, and find out whether the "
                "improvement paid for the turnover it added."
            ),
            "ml_lab": (
                "Compare the two models directly, with the train/test gap, the base rate and the "
                "per-regime breakdown side by side."
            ),
            "validation": (
                "Go past the single split: rolling walk-forward gives ten or more consecutive "
                "holdouts, which these exercises can only hint at."
            ),
            "backtest": (
                "Try new parameters — deciding your acceptance criterion before you look, which "
                "the parameter-sensitivity exercise is designed to make you care about."
            ),
            "learn": (
                "The eight-stage learning path, the three pitfalls, and every quant note these "
                "exercises reference."
            ),
        },
    },
    # ----------------------------------------------------------------------
    "learn": {
        "title": "Learn",
        "icon": ":material/menu_book:",
        "path": "app_pages/learn.py",
        "section": "Training",
        "step": None,
        "teaches": (
            "The reference layer: an eight-stage learning path, the three ways a backtest goes "
            "wrong, every quant note grouped by theme, and a glossary."
        ),
        "why": (
            "The other pages show you what happened. This one explains why the tools are built "
            "the way they are, and what to distrust about your own results."
        ),
        "habit": "Looking up the concept behind a confusing number, rather than moving past it.",
        "confused": {
            "note": "overfitting",
            "glossary": "The Glossary tab below defines every metric, strategy and feature.",
            "exercise": None,
        },
        "mistakes": [
            ("Reading the notes front to back",
             "They are reference, not a textbook. Read the one note that covers the number "
             "confusing you, then go back to the page you came from."),
            ("Treating the done-when conditions as tasks",
             "They describe a change in how you behave, not something you can complete by "
             "running code. You can finish every task here and satisfy none of them."),
            ("Reading the pitfalls only once, at the start",
             "They are far easier to recognise in work you have already done. Re-read them "
             "after your first result that looks good."),
        ],
        "next": ["backtest", "regimes", "adaptive", "ml_lab", "validation", "exercises"],
        "next_blurbs": {
            "backtest": (
                "Apply what you read about equity curves, drawdown and exposure on a real "
                "strategy, with every caveat firing on your own numbers."
            ),
            "regimes": (
                "Explore volatility clustering directly, and check the persistence that makes "
                "environments worth conditioning on at all."
            ),
            "adaptive": (
                "Test the four mechanisms against each other, and find out whether the simplest "
                "one captured most of the benefit."
            ),
            "ml_lab": (
                "See overfitting happen rather than read about it: toggle between the two models "
                "and watch the gap open up."
            ),
            "validation": (
                "Measure robustness properly — rolling folds, the environment-mix story, and a "
                "fair comparison of all eleven strategies."
            ),
            "exercises": (
                "Practise everything hands-on: ten exercises with automated checks against your "
                "own loaded data."
            ),
        },
    },
}


def breadcrumb(page_key: str) -> str:
    """'You are here' trail, e.g. 'Research › Backtest · step 1 of 6'."""
    guide = PAGE_GUIDE[page_key]
    parts = [guide["section"]] if guide["section"] else []
    parts.append(guide["title"])
    trail = " › ".join(parts)
    if guide["step"]:
        trail += f" · step {guide['step']} of {len(PAGE_ORDER)}"
    return trail
