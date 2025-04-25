import numpy as np
import pandas as pd
import streamlit as st
import bisect
# import numpy_financial as npf


version = "0.1.1"

## Todo:
# - Add a button to generate a PDF report of the analysis
# fix deductions and cash flow calculations
# add summary of property stats with ability to add images, address and notes


st.set_page_config(page_title="Property Investment Report", page_icon="🏠", layout="wide")
st.title("Property Investment Analysis")

with st.sidebar.form("inputs_form"):
    submitted = st.form_submit_button("Update")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Property", "Loan", "Expenses", "Income", "Depreciation", "Growth"])

    ## Property Inputs
    property_value = tab1.number_input("Property Value", 0, step=1000, value=635000)
    rental_income = tab1.number_input("Weekly Rent", 0, step=25, value=600)

    property_location = tab1.selectbox("Property Location", ["ACT", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"])
    
    property_type = tab1.selectbox("Property Type", ["House", "Unit", "Townhouse", "Apartment"])
    property_age = tab1.number_input("Year Built", 0, step=1, value=2025, format="%d")
    property_size = tab1.number_input("Property Size (sqm)", 0, step=1, value=300)
    property_bedrooms = tab1.slider("Number of Bedrooms", 0, step=1, value=3, max_value=10)
    property_bathrooms = tab1.slider("Number of Bathrooms", 0, step=1, value=2, max_value=5)
    property_garages = tab1.slider("Number of Garages", 0, step=1, value=1, max_value=5)

    ## Loan Inputs
    
    loan_amount_header = tab2.empty()

    # tab2.write()

    loan_type= tab2.radio("Loan Type", ["Interest Only", "Principal and Interest"])
    interest_rate = tab2.number_input("Interest rate (%)", 0.0, step=0.1, value=6.0, format="%.2f")

    equity_investment = tab2.number_input("Equity Investment", 0, step=1000, value=0)
    cash_investment = tab2.number_input("Cash Investment", 0, step=1000, value=0)

    tab2.subheader("Loan Expenses")
    loan_expense_header = tab2.empty()
    offset_account = tab2.number_input("Offset Account Fee", 0, step=50, value=0)
    lmi = tab2.number_input("Lenders Mortgage Insurance (LMI)", 0, step=1000, value=0)
    government_fees = tab2.number_input("Government Fees", 0, step=50, value=0)
    stamp_duty = tab2.number_input("Stamp Duty", 0, step=250, value=0)
    convayencor_fee = tab2.number_input("Conveyancing Fee", 0, step=50, value=1500)

    tab2.subheader("Online Calculators")
    tab2.link_button('Stamp Duty & LMI Calculator', 'https://www.westpac.com.au/personal-banking/home-loans/calculator/stamp-duty-calculator/')
    
    total_loan_expenses = offset_account + lmi + government_fees 
    

    loan_expense_header.write(f"**Total Loan Expenses:**   ${total_loan_expenses:,.0f}")

    loan_amount = property_value - cash_investment + stamp_duty + convayencor_fee + total_loan_expenses
    loan_amount_header.write(f"**Borrowed Amount:**   ${loan_amount:,.0f}")


    ## Expenses Inputs
    agent_commision = tab3.number_input("Agent Commission (%)", 0.0, step=0.1, value=7.0)
    letting_fee = tab3.number_input("Letting Fee", 0, step=10, value=1000)
    council_rates = tab3.number_input("Council Rates", 0, step=50, value=1500)
    insurance = tab3.number_input("Insurance", 0, step=50, value=1500)
    body_corp = tab3.number_input("Body Corporate", 0, step=100, value=0)
    land_tax = tab3.number_input("Land Tax", 0, step=100, value=2000)
    
    tab3.subheader("Online Calculators")
    tab3.link_button('Land Tax Calculator', 'https://www.e-business.sro.vic.gov.au/calculators/land-tax')

    ## Income Inputs
    partners_income_table = {
        "Partner": ["Partner 1", "Partner 2"],
        "Income": [
            100_000,
            67_000
        ], 
        "HECS": [
            True,
            True
        ]
    }
    partners_income_table_updated = tab4.data_editor(partners_income_table, 
                     num_rows='dynamic', 
                     hide_index=True, 
                     use_container_width=True, 
                    
                     column_config={
                         "Partner": st.column_config.TextColumn(),
                         "Income": st.column_config.NumberColumn(format="$ %0f"),
                         "HECS": st.column_config.CheckboxColumn()
                     }
    )

    ## Depreciation Inputs
    depreciation_schedule = {
        "Year": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "Depreciation": [
            depreciation_year1 := 13_000,
            depreciation_year2 := 10_000,
            depreciation_year3 := 9_000,
            depreciation_year4 := 8_000,
            depreciation_year5 := 8_000,
            depreciation_year6 := 7_000,
            depreciation_year7 := 7_000,
            depreciation_year8 := 6_000,
            depreciation_year9 := 6_000,
            depreciation_year10 := 6_000
        ]
    }
    updated_depreciation_schedule = tab5.data_editor(depreciation_schedule, 
                                                     num_rows='fixed', 
                                                     hide_index=True, 
                                                     use_container_width=True, 
                                                     disabled=['Year'],
                                                        column_config={
                                                            "Year": st.column_config.NumberColumn(format="%.0f"),
                                                            "Depreciation": st.column_config.NumberColumn(format="$ %0f")
                                                        }
    )

    tab5.link_button("Depreciation Calculator", "https://www.washingtonbrown.com.au/depreciation/calculator/", 
                     help="Calculate depreciation for your property")
    


    inflation_rate = tab6.number_input("Inflation rate (%)", 0.0, step=0.1, value=3.0)
    rental_income_growth = tab6.number_input("Rental Income Growth (%)", 0.0, step=0.1, value=10.0)
    property_growth = tab6.number_input("Capital Growth (%)", 0.0, step=0.1, value=5.0)
    vacancy_rate = tab6.number_input("Vacancy Rate (%)", 0.0, step=0.1, value=1.0)

    #footer

    st.markdown("""---""")
    st.markdown(f"Version {version}")


if st.button("📄 Generate PDF"):
    pass

# summary
coll, col1, col2, colr = st.columns(4)
col1.subheader("Assumptions")
col1.metric("Property Value", f"${property_value:,.0f}")
col1.metric('Loan Expenses', f"${total_loan_expenses:,.0f}")
col1.metric("Loan Amount", f"${loan_amount:,.0f}")
col1.metric("Rental Income", f"${rental_income:,.0f}")

col2.subheader("Projection")
col2.metric("Gross Rental Yield", f"{round((rental_income*52) / property_value * 100,2)}%")
col2.metric("Property Value (10 years)", f"${property_value * (1 + property_growth / 100) ** 9:,.0f}")
col2.metric("Equity in Property (10 years)", f"${property_value * (1 + property_growth / 100) ** 9 - loan_amount:,.0f}")


# 10 year projection df table
years = np.arange(1, 11)
rental_income_projection = np.zeros(10)
# rental_income_projection[0] = (rental_income / 7) * 365
rental_income_projection[0] = rental_income

for i in range(1, 10):
    rental_income_projection[i] = rental_income_projection[i-1] * (1 + rental_income_growth / 100) #* (1 - vacancy_rate / 100)

annual_rental_income = rental_income_projection * 52 * (1 - vacancy_rate / 100)
annual_agent_commision = (agent_commision/100) *  annual_rental_income
annual_expenses = annual_agent_commision + council_rates + insurance + body_corp + land_tax

projection_df = pd.DataFrame({
    "Year": np.arange(1, 11),
    "Property Value": property_value * (1 + property_growth / 100) ** np.arange(0, 10),
    "Weekly Rent": rental_income_projection,
    "Annual Rental Income": annual_rental_income,
    "Depreciation": updated_depreciation_schedule["Depreciation"],
    "Expenses": annual_expenses
})

# ((rental_income_projection[i]*52) * (1 - vacancy_rate / 100))

projection_df["Property Value"] = projection_df["Property Value"]
projection_df["Annual Rental Income"] = projection_df["Annual Rental Income"]
projection_df['Interest'] = interest_rate/100 * loan_amount
projection_df["Depreciation"] = projection_df["Depreciation"]
projection_df["Expenses"] = projection_df["Expenses"]

st.markdown("---")

st.subheader("10 Year Projection")

st.dataframe(projection_df, use_container_width=True, hide_index=True,
             column_config={
                "Year": st.column_config.NumberColumn(format="%.0f"),
                "Property Value": st.column_config.NumberColumn(format="$ %.0f"),
                "Weekly Rent": st.column_config.NumberColumn(format="$ %.0f"),
                "Annual Rental Income": st.column_config.NumberColumn(format="$ %.0f"),
                "Depreciation": st.column_config.NumberColumn(format="$ %.0f"),
                "Expenses": st.column_config.NumberColumn(format="$ %.0f"),
                "Interest": st.column_config.NumberColumn(format="$ %.0f")
            }
)

st.subheader("Deductions & Cash Flow")

loan_cost_split = np.zeros(10)
for i in range(5):
    loan_cost_split[i] = total_loan_expenses / 5


cash_flow_df = pd.DataFrame({
    "Year": np.arange(1, 11),
    "Equity": projection_df["Property Value"] - loan_amount,
    "Pre Tax Cash Flow": projection_df["Annual Rental Income"] - projection_df["Expenses"] - projection_df["Interest"],
    "Total Deductions": projection_df["Depreciation"] + projection_df["Expenses"] + projection_df["Interest"] + loan_cost_split,
    "Tax Credit": 0,  # Placeholder for tax credit calculation
    "After Tax Cash Flow": 0,  # Placeholder for after tax cash flow calculation,
    "Income per Week": 0  # Placeholder for income per week calculation
})


# equity (property value - loan amount)
# pre tax cash flow ( rental income - expenses - interest)
# total deductions ( loan costs + depreciation + expenses + interest)
# tax credit (sum of difference in tax paid for partners)
# after tax cash flow (tax credit +pre tax cash flow)
# income per week (after-tax cash flow / 52)


def calculate_tax(x, b=[0, 18_200, 45_000, 135_000, 190_000, float('inf')],
                  r=[0, .16, .30, .37, .45]):
    return sum((min(x, b[i+1]) - b[i]) * r[i] for i in range(len(r)) if x > b[i])

hecs_rates = {
    0:      0.01,
    54_435: 0.01,
    62_851: 0.02,
    66_621: 0.025,
    70_619: 0.03,
    74_856: 0.035,
    79_347: 0.04,
    84_108: 0.045,
    89_155: 0.05,
    94_504: 0.055,
    100_175: 0.06,
    106_186: 0.065,
    112_557: 0.07,
    119_310: 0.075,
    126_468: 0.08,
    134_057: 0.085,
    142_101: 0.09,
    150_627: 0.095,
    159_664: 0.1,
}



THRESHOLDS = sorted(hecs_rates) 
RATES       = [hecs_rates[t] for t in THRESHOLDS]

def calculate_hecs(income, thresholds=THRESHOLDS, rates=RATES):
    idx = bisect.bisect_right(thresholds, income) - 1
    hecs_paid = rates[max(idx, 0)] * income
    return hecs_paid

LITO_TABLE = {
    0: (700, 0.0),
    37_500: (700, 0.05),
    45_000: (325, 0.015),
    66_667: (0.0, 0.0)
}

LITO_THRESHOLDS = sorted(LITO_TABLE)  
LITO_BASES, LITO_TAPERS = zip(*(LITO_TABLE[t] for t in LITO_THRESHOLDS))

def calculate_lito(income, thresholds=LITO_THRESHOLDS, bases=LITO_BASES, tapers=LITO_TAPERS):
    """Return the Low-Income Tax Offset for a given `income`."""
    idx = bisect.bisect_right(thresholds, income) - 1         # find bracket
    base, taper, floor = bases[idx], tapers[idx], thresholds[idx]
    return max(base - (income - floor) * taper, 0.0)

tax_credit = 0


df_temp = pd.DataFrame(partners_income_table_updated)
n_partners = df_temp["Partner"].nunique()

for i in range(10):
    
    tax_credit = 0
    for partner in df_temp['Partner']:
        partners_df = df_temp[df_temp['Partner'] == partner].copy()

        income = partners_df["Income"].values[0]
        new_income = partners_df["Income"].values[0] + (annual_rental_income[i] / n_partners) - (cash_flow_df["Total Deductions"][i] / n_partners)

        hecs = 0
        if partners_df["HECS"].values[0] == True:
            new_hecs = calculate_hecs(new_income)
            current_hecs = calculate_hecs(income)

        current_medicare = income * 0.02
        new_medicare = new_income * 0.02

        current_tax = calculate_tax(income) + current_hecs - calculate_lito(income) + current_medicare
        new_tax = calculate_tax(new_income) + new_hecs - calculate_lito(new_income) + new_medicare

        tax_difference = current_tax - new_tax
        tax_credit += tax_difference

    cash_flow_df["Tax Credit"][i] += tax_credit
    cash_flow_df["After Tax Cash Flow"][i] += (cash_flow_df["Pre Tax Cash Flow"][i] + tax_credit)
    cash_flow_df["Income per Week"][i] += cash_flow_df["After Tax Cash Flow"][i] / 52




st.dataframe(cash_flow_df, use_container_width=True, hide_index=True,
             column_config={
                "Year": st.column_config.NumberColumn(format="%.0f"),
                "Equity": st.column_config.NumberColumn(format="$ %.0f"),
                "Pre Tax Cash Flow": st.column_config.NumberColumn(format="$ %.0f"),
                "Total Deductions": st.column_config.NumberColumn(format="$ %.0f"),
                "Tax Credit": st.column_config.NumberColumn(format="$ %.0f"),
                "After Tax Cash Flow": st.column_config.NumberColumn(format="$ %.0f"),
                "Income per Week": st.column_config.NumberColumn(format="$ %.0f")
                }
)
st.markdown("---")

