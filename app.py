import numpy as np
import pandas as pd
import streamlit as st
import bisect
from streamlit_searchbox import st_searchbox
import requests


version = "1.0.1"

## Todo:
# - Add a button to generate a PDF report of the analysis
# fix deductions and cash flow calculations
# add summary of property stats with ability to add images, address and notes


st.set_page_config(page_title="Property Investment Report", page_icon="🏠", layout="wide")
st.title("Property Investment Report")
st.markdown("###")

GEO_KEY = st.secrets["GEOAPIFY_KEY"]

def geoapify_suggest(q: str):
    if len(q) < 5:
        return []
    params = {
        "text": q,
        "limit": 5,
        "filter": "countrycode:au",   
        "format": "json",
        "apiKey": GEO_KEY,
    }
    r = requests.get(
        "https://api.geoapify.com/v1/geocode/autocomplete",
        params=params,
        timeout=4,
    )
    r.raise_for_status()
    return [
        (f["formatted"], f["formatted"])
        for f in r.json().get("results", [])
    ]

address = st_searchbox(geoapify_suggest, placeholder="Enter address")


with st.sidebar.form("inputs_form"):
    
    submitted = st.form_submit_button("Update")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Property", "Loan", "Expenses", "Income", "Depreciation", "Growth"])

    ## Property Inputs
    with tab1:
        property_image = st.file_uploader("Upload Property Image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

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


from fpdf import FPDF
import base64

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

col1, col2, _, _, _, _, _, _ = st.columns(8)
# download_pdf = col1.button("📄 Generate PDF")
download_holder = col1.empty()


if address is not None:
    st.markdown(f"""## {address}""")
else:
    st.markdown(f"""## No Address Provided""")

# TODO: add property stats
col1, col2, _, _, _ = st.columns(5)
col1.metric("Property Type", property_type)
col1.metric("Property Age", property_age)
col1.metric("Property Size", f"{property_size} sqm")
col2.metric("Bedrooms", property_bedrooms)
col2.metric("Bathrooms", property_bathrooms)
col2.metric("Garages", property_garages)



# show grid of multiple images at fixed size responsive to number of images uploaded
if property_image is not None:
    cols = st.columns(3)
    for i, img in enumerate(property_image):
        with cols[i % 3]:
            st.image(img, width=300, caption=f"Image {i+1}")


notes = st.text_area("Additional Notes", placeholder="Add any additional notes here...", height=(34*4))

# summary

st.markdown("---")
st.subheader("Summary")
coll, col1, col2, colr = st.columns(4)

col1.metric("Property Value", f"${property_value:,.0f}")
col1.metric('Loan Expenses', f"${total_loan_expenses:,.0f}")
col1.metric("Loan Amount", f"${loan_amount:,.0f}")
col1.metric("Rental Income", f"${rental_income:,.0f}")


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
                "Year": st.column_config.NumberColumn(format="%.0f", pinned=True),
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
                "Year": st.column_config.NumberColumn(format="%.0f", pinned=True),
                "Equity": st.column_config.NumberColumn(format="$ %.0f"),
                "Pre Tax Cash Flow": st.column_config.NumberColumn(format="$ %.0f"),
                "Total Deductions": st.column_config.NumberColumn(format="$ %.0f"),
                "Tax Credit": st.column_config.NumberColumn(format="$ %.0f"),
                "After Tax Cash Flow": st.column_config.NumberColumn(format="$ %.0f"),
                "Income per Week": st.column_config.NumberColumn(format="$ %.0f")
                }
)
st.markdown("---")




# --- PDF helpers -------------------------------------------------------------
from fpdf import FPDF
from io import BytesIO
import os
from datetime import datetime
import zoneinfo      

MEL_TZ = zoneinfo.ZoneInfo("Australia/Melbourne")

class PDF(FPDF):
    def header(self):
        # Title bar on every page
        self.set_fill_color(230, 230, 230)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, "Property Investment Report", 0, 1, "C", fill=True)
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)

        date_str = datetime.now(MEL_TZ).strftime("Generated %d-%m-%Y %H:%M")
        self.cell(0, 8, date_str, 0, 0, "L")

        # -- page number on the right --------------------------------------------
        self.cell(0, 8, f"Page {self.page_no()}", 0, 0, "R")


def add_key_metrics(pdf, metrics: dict):
    """
    Render key-value pairs in two columns.

    metrics = {"Property Value": "$635,000", "Weekly Rent": "$600", ...}
    """
    pdf.set_font("Helvetica", "", 11)
    
    # full printable width (left + right margins already excluded)
    page_w = pdf.w - 2 * pdf.l_margin
    col_pair_w = page_w / 2          # width for one “label + value” pair
    label_w   = col_pair_w * 0.55    # 55 % label | 45 % value
    value_w   = col_pair_w - label_w

    for idx, (k, v) in enumerate(metrics.items()):
        # label + value for this metric
        pdf.cell(label_w, 6, k + ":", 0, 0)
        pdf.cell(value_w, 6, v,      0, 0)

        # after every 2nd metric, move to next line
        if idx % 2 == 1:
            pdf.ln(6)

    # if odd number of metrics, make sure we end the row cleanly
    if len(metrics) % 2 == 1:
        pdf.ln(6)

    pdf.ln(2)     # small spacer below the whole block


def df_to_table(pdf, df, title):
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, title, 0, 1)
    pdf.set_font("Helvetica", "", 8)

    # Column widths proportional to page width
    col_w = (pdf.w - 2*pdf.l_margin) / len(df.columns)
    pdf.set_fill_color(240, 240, 240)

    # Header row
    for col in df.columns:
        pdf.cell(col_w, 6, str(col), border=1, align="C", fill=True)
    pdf.ln()

    # Data rows
    for _, row in df.iterrows():
        for col, item in row.items():
            if isinstance(item, (int, float)) and col != "Year":
                txt = f"${item:,.0f}"          # $ + comma-separated, no decimals
            else:
                txt = str(item)
            pdf.cell(col_w, 6, txt, border=1, align="C")
        pdf.ln()



def upload_to_stream(uploaded):
    """Return (BytesIO stream, image_type) for an st.uploaded_file"""
    stream = BytesIO(uploaded.getvalue())   # raw bytes → stream
    stream.seek(0)
    ext = os.path.splitext(uploaded.name)[1].lower()
    img_type = "PNG" if ext == ".png" else "JPEG"   # default to JPEG
    return stream, img_type

import tempfile, os, contextlib

@contextlib.contextmanager
def uploaded_to_tempfile(uploaded):
    """
    Yields the name of a temporary image file created from a Streamlit
    UploadedFile.  The file is removed automatically afterwards.
    """
    ext = os.path.splitext(uploaded.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded.getvalue())
        tmp.flush()
        tmp_name = tmp.name       # keep path before closing
    try:
        yield tmp_name            # hand the path back to caller
    finally:
        os.remove(tmp_name)       # clean up


def generate_pdf(address,
                 metrics_dict,
                 home_metrics,
                 projection_df,
                 cash_flow_df,
                 notes_text,
                 images):
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()


    # First image (if any) centred under title
    if images:
        with uploaded_to_tempfile(images[0]) as img_path:
            pdf.image(img_path, x=(pdf.w/2 - 40), w=80)
        pdf.ln(2)

    # Cover section -----------------------------------------------------------
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, address or "No Address Provided", 0, 1)
    pdf.ln(2)

    # Summary metrics ---------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    # pdf.cell(0, 8, "Key Metrics", 0, 1)
    add_key_metrics(pdf, home_metrics)
    pdf.ln(3)

    # Summary metrics ---------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Key Metrics", 0, 1)
    add_key_metrics(pdf, metrics_dict)
    pdf.ln(3)
    

    # Notes -------------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Notes", 0, 1)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, notes_text or "—")

    pdf.ln(2)
    pdf.add_page()
    # Projection table --------------------------------------------------------
    df_to_table(pdf, projection_df, "10-Year Projection")

    # Cash-flow table ---------------------------------------------------------
    df_to_table(pdf, cash_flow_df, "Deductions & Cash Flow")


    # extra photos
    if len(images) > 1:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Property Photos", 0, 1)
        pdf.ln(2)

        w = (pdf.w - 2*pdf.l_margin - 10) / 2
        h = w * 0.75
        for idx, upl in enumerate(images[1:], start=1):
            with uploaded_to_tempfile(upl) as img_path:
                pdf.image(img_path, w=w, h=h)
            if idx % 2 == 0:
                pdf.ln(h + 4)
            else:
                pdf.set_x(pdf.l_margin + w + 10)

    # Return bytes
    return pdf.output(dest="S").encode("latin-1")



@st.cache_resource(show_spinner="Building report…")
def build_report(**kwargs):
    return generate_pdf(**kwargs)

if address is not None:
    # address = address.replace(" ","").replace(",","_")
    # address = address.replace(" ","_").replace(",","_")
    address_safe = address.replace(" ","").replace(",","_")
else:
    address_safe = "No_Address_Provided"
download_holder.download_button(label=":material/download: PDF Report",
                                data=build_report(
                                                    address=address,
                                                    metrics_dict={
                                                        "Property Value": f"${property_value:,.0f}",
                                                        "Weekly Rent": f"${rental_income:,.0f}",
                                                        "Loan Amount": f"${loan_amount:,.0f}",
                                                        "Gross Rental Yield": f"{round((rental_income*52)/property_value*100,2)}%",
                                                        "Property Value (10 years)": f"${property_value * (1 + property_growth / 100) ** 9:,.0f}",
                                                        "Equity in Property (10 years)": f"${property_value * (1 + property_growth / 100) ** 9 - loan_amount:,.0f}",
                                                    },
                                                    home_metrics={
                                                        "Property Type": f"{property_type}",
                                                        "Property Size": f"{property_size} sqm",
                                                        "Build Year": f"{property_age}",
                                                        "Bedrooms": f"{property_bedrooms}",
                                                        "Bathrooms": f"{property_bathrooms}",
                                                        "Car Spaces": f"{property_garages}"
                                                    },
                                                    projection_df=projection_df.round(0).astype(int),
                                                    cash_flow_df=cash_flow_df.round(0).astype(int),
                                                    notes_text=notes,
                                                    images=property_image,         
                                                ),
                                file_name=f"{address_safe}.pdf",
                                mime="application/pdf",
                                help="Download the report as a PDF file"
            )

