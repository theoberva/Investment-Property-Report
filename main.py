from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import List, Sequence

import numpy as np
import numpy_financial as npf
import pandas as pd
import streamlit as st
from jinja2 import BaseLoader, Environment, select_autoescape
from weasyprint import HTML

###############################################################################
# Helpers & Dataclass
###############################################################################

FMT_CURRENCY = lambda v: f"${v:,.0f}"
FMT_PCT = lambda v: f"{v * 100:.2f}%" if v < 1 else f"{v:.2f}%"


def _pct(x: float) -> float:  # convert 8 → 0.08 if user enters whole number
    return x / 100 if x > 1 else x


@dataclass
class Inputs:
    # ‑‑ Purchase & loan ------------------------------------------------------
    purchase_price: float = 635_000
    renovation_costs: float = 0
    conveyancing: float = 1_500
    stamp_duty: float = 10_000
    initial_investment: float = 155_000  # upfront cash/equity

    loan_amount: float = 497_342
    interest_rate: float = 0.06  # decimal
    loan_term: int = 30  # yrs
    io_years: int = 30  # interest‑only period
    loan_costs: float = 5_842  # establishment etc.

    # ‑‑ Macro assumptions ----------------------------------------------------
    cap_growth: float = 0.08  # decimal
    inflation: float = 0.03

    # ‑‑ Rent & operating -----------------------------------------------------
    weekly_rent: float = 630
    vacancy: float = 0.0  # decimal
    expenses_pct: float = 0.1916  # of gross rent

    # ‑‑ Tax & depreciation ---------------------------------------------------
    taxable_income: float = 157_000  # combined (for credit estimate)
    marginal_tax: float = 0.325  # decimal after offsets (rough for 90‑120k)
    building_cost: float = 310_000
    build_depr: float = 0.025
    fittings_value: float = 55_000
    fittings_depr_yr1: float = 10_513  # diminishing value year‑1

    # ‑‑ Exit / selling assumptions ------------------------------------------
    selling_commission_pct: float = 0.0225  # 2.25% + GST typical
    solicitor_fee_base: float = 1_500
    cgt_discount: float = 0.50  # 50% CGT discount after 12m held (AU)

    # ‑‑ Horizon --------------------------------------------------------------
    horizon: int = 10
    projection_years: Sequence[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])

    def gross_rent_year1(self):
        return self.weekly_rent * 52

###############################################################################
# Core Calculations
###############################################################################

def build_projection(inp: Inputs) -> pd.DataFrame:
    rng = range(0, inp.horizon + 1)
    df = pd.DataFrame(index=rng)

    # Property & equity -------------------------------------------------------
    df["property_value"] = inp.purchase_price * (1 + inp.cap_growth) ** df.index
    df["loan_balance"] = inp.loan_amount  # interest‑only → constant
    df["equity"] = df["property_value"] - df["loan_balance"]

    # Economic rates (for display) -------------------------------------------
    df["cap_growth_rate"] = inp.cap_growth
    df["inflation_rate"] = inp.inflation

    # Rent & expenses ---------------------------------------------------------
    df["gross_rent_week"] = inp.weekly_rent * (1 + inp.inflation) ** df.index
    df["gross_rent"] = df["gross_rent_week"] * 52
    df["expenses"] = df["gross_rent"] * inp.expenses_pct

    # Interest (IO) -----------------------------------------------------------
    df["interest"] = inp.loan_amount * inp.interest_rate

    # Cash flow ---------------------------------------------------------------
    df["pre_tax_cf"] = (
        df["gross_rent"] * (1 - inp.vacancy) - df["expenses"] - df["interest"]
    )

    # Depreciation / loan‑cost deductions ------------------------------------
    df["depr_build"] = inp.building_cost * inp.build_depr
    df["depr_fit"] = inp.fittings_depr_yr1 / (1.5 ** (df.index - 1))
    df["loan_cost_ded"] = np.where(df.index <= 5, inp.loan_costs / 5, 0)

    df["total_ded"] = (
        df["interest"] + df["expenses"] + df["depr_build"] + df["depr_fit"] + df["loan_cost_ded"]
    )
    df["tax_credit"] = df["total_ded"] * inp.marginal_tax
    df["after_tax_cf"] = df["pre_tax_cf"] + df["tax_credit"]

    return df


def equity_after_sale(inp: Inputs, df: pd.DataFrame) -> dict:
    """Compute sale‑year equity net of costs & CGT."""
    pv_end = df["property_value"].iloc[-1]
    loan = df["loan_balance"].iloc[-1]
    equity_gross = pv_end - loan

    # Selling costs
    commission = pv_end * inp.selling_commission_pct
    solicitor = inp.solicitor_fee_base * (pv_end / inp.purchase_price) ** 0.5  # scale roughly

    # Capital Gain Tax (simplistic – ignores cost‑base adjustments)
    gain = pv_end - inp.purchase_price - inp.renovation_costs
    taxable_gain = gain * inp.cgt_discount
    cgt = taxable_gain * inp.marginal_tax

    after_sale_equity = equity_gross - commission - solicitor - cgt
    return dict(gross=equity_gross, commission=commission, solicitor=solicitor, cgt=cgt, net=after_sale_equity)


def irr_and_npv(inp: Inputs, df: pd.DataFrame, exit_equity_net: float) -> tuple[float, float]:
    cashflows = [-inp.initial_investment] + list(df["after_tax_cf"])
    cashflows[-1] += exit_equity_net  # realise equity at sale yr
    irr = npf.irr(cashflows)
    npv = npf.npv(0.10, cashflows[1:])  # 10% discount for display
    return irr, npv

###############################################################################
# PDF Rendering
###############################################################################

HTML_TEMPLATE = """
<!doctype html><html><head><meta charset='utf-8'>
<style>
body{font-family:Arial,Helvetica,sans-serif;font-size:10pt;margin:2cm}
h1,h2{color:#003366;margin-bottom:0.2cm}
table{width:100%;border-collapse:collapse;margin-bottom:0.7cm}
th,td{border:1px solid #ccc;padding:4px 6px;text-align:right}
th{background:#f0f6f8}
.left{text-align:left}
.small{font-size:8pt}
</style></head><body>
<h1>PROPERTY INVESTMENT ANALYSIS (DESCRIPTIVE)</h1>
<p><strong>Date prepared:</strong> {{ today }}</p>

<h2>SUMMARY</h2>
<table>
<tr><th class='left'>Assumptions</th><th>Value</th><th class='left'>Results ({{ inp.horizon }} yrs)</th><th>Value</th></tr>
<tr><td class='left'>Property value</td><td>{{ inp.purchase_price | cur }}</td><td class='left'>Property value</td><td>{{ df.property_value.iloc[-1] | cur }}</td></tr>
<tr><td class='left'>Initial investment</td><td>{{ inp.initial_investment | cur }}</td><td class='left'>Equity</td><td>{{ df.equity.iloc[-1] | cur }}</td></tr>
<tr><td class='left'>Gross rental yield (yr 1)</td><td>{{ gross_yield | pct }}</td><td class='left'>After‑tax return /yr (IRR)</td><td>{{ irr | pct }}</td></tr>
<tr><td class='left'>Net rental yield (yr 1)</td><td>{{ net_yield | pct }}</td><td class='left'>Net present value (10%)</td><td>{{ npv | cur }}</td></tr>
<tr><td class='left'>Capital growth rate</td><td>{{ inp.cap_growth | pct }}</td><td class='left'>IF SOLD<br>(selling costs & CGT)</td><td>{{ sell_costs | cur }}</td></tr>
<tr><td class='left'>Inflation rate</td><td>{{ inp.inflation | pct }}</td><td class='left'>Equity after sale</td><td>{{ after_sale.net | cur }}</td></tr>
</table>

<h2>PROJECTIONS</h2>
<table>
<thead><tr><th class='left'>Year</th>{% for c in proj_cols %}<th>{{ c.replace('_',' ').title() }}</th>{% endfor %}</tr></thead>
<tbody>
{% for yr,row in proj_tbl.iterrows() %}<tr><td class='left'>{{ yr }}</td>{% for c in proj_cols %}<td>{{ row[c]|cur if c not in pct_cols else row[c]|pct }}</td>{% endfor %}</tr>{% endfor %}
</tbody>
</table>

<div style='page-break-before:always'></div>
<h2>DETAILED NOTES</h2>
<h3>Purchase Costs</h3>
<table>
<tr><td class='left'>Conveyancing fees</td><td>{{ inp.conveyancing|cur }}</td></tr>
<tr><td class='left'>Stamp duty</td><td>{{ inp.stamp_duty|cur }}</td></tr>
<tr><th class='left'>Total purchase costs</th><th>{{ purchase_total|cur }}</th></tr>
</table>

<h3>Investment &amp; Loan</h3>
<table>
<tr><th class='left'></th><th>Investments</th><th>Loan</th><th>Total cost</th></tr>
<tr><td class='left'>Property costs</td><td>{{ inp.initial_investment|cur }}</td><td>{{ inp.loan_amount|cur }}</td><td>{{ inp.purchase_price|cur }}</td></tr>
<tr><td class='left'>Purchase costs</td><td>0</td><td>{{ purchase_total|cur }}</td><td>{{ purchase_total|cur }}</td></tr>
<tr><td class='left'>Loan costs</td><td>0</td><td>{{ inp.loan_costs|cur }}</td><td>{{ inp.loan_costs|cur }}</td></tr>
<tr><th class='left'>Totals</th><th>{{ inp.initial_investment|cur }}</th><th>{{ inp.loan_amount + purchase_total + inp.loan_costs|cur }}</th><th>{{ inp.initial_investment + inp.loan_amount + purchase_total + inp.loan_costs|cur }}</th></tr>
</table>

<h3>Equity Projection &amp; Sale Costs</h3>
<table>
<tr><th class='left'>Item</th><th>Value</th></tr>
<tr><td class='left'>Equity (gross)</td><td>{{ after_sale.gross|cur }}</td></tr>
<tr><td class='left'>Sales commission</td><td>{{ after_sale.commission|cur }}</td></tr>
<tr><td class='left'>Solicitor fees</td><td>{{ after_sale.solicitor|cur }}</td></tr>
<tr><td class='left'>Capital gains tax</td><td>{{ after_sale.cgt|cur }}</td></tr>
<tr><th class='left'>Equity (after sale)</th><th>{{ after_sale.net|cur }}</th></tr>
</table>

<h3>Interest Costs &amp; Loan Type</h3>
<table>
<tr><td class='left'>Loan type</td><td>Interest‑only</td></tr>
<tr><td class='left'>Interest rate</td><td>{{ inp.interest_rate|pct }}</td></tr>
<tr><td class='left'>Monthly payment</td><td>{{ monthly_payment|cur }}</td></tr>
</table>

<h3>Rent &amp; Annual Expenses (Year 1)</h3>
<table>
<tr><td class='left'>Potential annual rent</td><td>{{ rent_yr1|cur }}</td></tr>
<tr><td class='left'>Vacancy rate</td><td>{{ inp.vacancy|pct }}</td></tr>
<tr><td class='left'>Actual annual rent</td><td>{{ rent_yr1_actual|cur }}</td></tr>
<tr><td class='left'>Operating expenses (% of rent)</td><td>{{ inp.expenses_pct|pct }}</td></tr>
<tr><td class='left'>Operating expenses</td><td>{{ expenses_yr1|cur }}</td></tr>
</table>

<h3>Depreciation Schedule (Year 1)</h3>
<table>
<tr><td class='left'>Building ({{ inp.build_depr|pct }} of {{ inp.building_cost|cur }})</td><td>{{ depr_build|cur }}</td></tr>
<tr><td class='left'>Fittings</td><td>{{ depr_fittings|cur }}</td></tr>
</table>

<h3>Total Tax Deductions (Year 1)</h3>
<table>
<tr><td class='left'>Interest</td><td>{{ interest_yr1|cur }}</td></tr>
<tr><td class='left'>Expenses</td><td>{{ expenses_yr1|cur }}</td></tr>
<tr><td class='left'>Depreciation (building)</td><td>{{ depr_build|cur }}</td></tr>
<tr><td class='left'>Depreciation (fittings)</td><td>{{ depr_fittings|cur }}</td></tr>
<tr><td class='left'>Loan costs amortised</td><td>{{ loan_cost_amort|cur }}</td></tr>
<tr><th class='left'>Total deductions</th><th>{{ total_ded_yr1|cur }}</th></tr>
<tr><td class='left'>Tax credit ({{ inp.marginal_tax|pct }} rate)</td><td>{{ tax_credit_yr1|cur }}</td></tr>
</table>

<p class='small'>Disclaimer: The projections listed above simply illustrate the outcome calculated from the input values and the assumptions contained in the model. Hence the figures can be varied as required and are in no way intended to be a guarantee of future performance. Although the information is provided in good faith, it is also given on the basis that no person using the information, in whole or in part, shall have any claim against the author or its servants, employees or consultants. This information is intended as general advice only and does not take account of individual needs or financial circumstances.</p>
</body></html>
"""


def render_pdf(inp: Inputs, df: pd.DataFrame) -> bytes:
    after_sale = equity_after_sale(inp, df)
    irr, npv_val = irr_and_npv(inp, df, after_sale["net"])

    env = Environment(loader=BaseLoader(), autoescape=select_autoescape())
    env.filters.update(cur=FMT_CURRENCY, pct=FMT_PCT)

    html = env.from_string(HTML_TEMPLATE).render(
        today=date.today().isoformat(), inp=inp, df=df,
        proj_tbl=df.loc[inp.projection_years], proj_cols=["property_value", "loan_balance", "equity", "gross_rent", "expenses", "after_tax_cf"],
        pct_cols=["cap_growth_rate", "inflation_rate"],
        gross_yield=(inp.gross_rent_year1() / inp.purchase_price),
        net_yield=(inp.gross_rent_year1()*(1-inp.expenses_pct) / inp.purchase_price),
        irr=irr, npv=npv_val,
        sell_costs=after_sale["commission"] + after_sale["solicitor"] + after_sale["cgt"],
        after_sale=after_sale,
        purchase_total=inp.conveyancing + inp.stamp_duty,
        monthly_payment= (inp.loan_amount * inp.interest_rate) /12,
        rent_yr1=inp.gross_rent_year1(),
        rent_yr1_actual=inp.gross_rent_year1()*(1-inp.vacancy),
        expenses_yr1=inp.gross_rent_year1()*inp.expenses_pct,
        depr_build=inp.building_cost*inp.build_depr,
        depr_fittings=inp.fittings_depr_yr1,
        loan_cost_amort=inp.loan_costs/5,
        interest_yr1=inp.loan_amount*inp.interest_rate,
        total_ded_yr1=(inp.loan_amount*inp.interest_rate)+inp.gross_rent_year1()*inp.expenses_pct+(inp.building_cost*inp.build_depr)+inp.fittings_depr_yr1+(inp.loan_costs/5),
        tax_credit_yr1=((inp.loan_amount*inp.interest_rate)+inp.gross_rent_year1()*inp.expenses_pct+(inp.building_cost*inp.build_depr)+inp.fittings_depr_yr1+(inp.loan_costs/5))*inp.marginal_tax,
    )
    return HTML(string=html).write_pdf()

###############################################################################
# Streamlit UI
###############################################################################

st.set_page_config(page_title="Property Investment Report", page_icon="💰", layout="wide")
st.title("🏠 Property Investment Analysis")

with st.sidebar.form("inputs_form"):
    st.subheader("Purchase & Loan")
    purchase_price = st.number_input("Purchase price", 0, step=1000, value=635000)
    renovation = st.number_input("Renovation costs", 0, step=1000, value=0)
    initial_inv = st.number_input("Initial cash investment", 0, step=1000, value=155000)
    # loan_amount = st.number_input("Loan amount", 0, step=1000, value=497342)
    int_rate = st.number_input("Interest rate (%)", 0.0, 15.0, value=6.0)
    loan_costs = st.number_input("Loan costs", 0, step=100, value=5842)
    conveyancing = st.number_input("Conveyancing fees", 0, step=100, value=1500)
    stamp_duty = st.number_input("Stamp duty", 0, step=100, value=10000)

    st.subheader("Market assumptions")
    cap = st.number_input("Capital growth (%)", 0.0, 20.0, value=5.0)
    infl = st.number_input("Inflation (%)", 0.0, 10.0, value=3.0)

    st.subheader("Rent & Expenses")
    rent = st.number_input("Weekly rent", 0, step=10, value=550)
    vacancy = st.number_input("Vacancy rate (%)", 0.0, 100.0, value=0.0)
    expenses_pct = st.number_input("Operating expenses (% of rent)", 0.0, 100.0, value=19.16)

    st.subheader("Tax & Depreciation")
    marginal_tax = st.number_input("Marginal tax rate (%)", 0.0, 50.0, value=32.5)

    horizon = st.slider("Projection horizon (yrs)", 1, 30, 10)
    submitted = st.form_submit_button("🔄  Update inputs")


if submitted or "inp" not in st.session_state:
    st.session_state.inp = Inputs(
        purchase_price=purchase_price,
        renovation_costs=renovation,
        conveyancing=conveyancing,
        stamp_duty=stamp_duty,
        initial_investment=initial_inv,
        loan_amount=purchase_price - initial_inv + renovation + stamp_duty + conveyancing,
        interest_rate=_pct(int_rate),
        loan_costs=loan_costs,
        cap_growth=_pct(cap),
        inflation=_pct(infl),
        weekly_rent=rent,
        vacancy=_pct(vacancy),
        expenses_pct=_pct(expenses_pct),
        marginal_tax=_pct(marginal_tax),
        horizon=horizon,
    )

inp: Inputs = st.session_state.inp

# Build projection & show snapshot
proj_df = build_projection(inp)

st.subheader("Projection snapshot at key years")
st.dataframe(proj_df.loc[inp.projection_years].style.format("${:,.0f}"))

if st.button("📄 Generate PDF"):
    pdf = render_pdf(inp, proj_df)
    st.success("Report created!")
    st.download_button(
        "⬇️ Download report",
        data=pdf,
        file_name="investment_report.pdf",
        mime="application/pdf",
    )
