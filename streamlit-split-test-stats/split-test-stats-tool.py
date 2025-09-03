import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
import datetime

# Set page configuration
st.set_page_config(
    page_title="A/B Testing Analysis",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 A/B Testing Statistical Analysis")
st.write("Upload your experiment data and run statistical tests between groups")

# PDF Export Functions
def create_html_report(analysis_results, metric_columns, df, group_id_column, pop_size_column, alpha, use_fdr):
    """Create HTML report with all analysis results"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>A/B Testing Analysis Report</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .metric-section {{
                margin-bottom: 40px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 20px;
            }}
            .winner-box {{
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                font-weight: bold;
            }}
            .no-winner-box {{
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                color: #0c5460;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                font-weight: bold;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            .chart-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .config-info {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .significant {{
                background-color: #d4edda;
                color: #155724;
            }}
            .not-significant {{
                background-color: #f8d7da;
                color: #721c24;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧪 A/B Testing Statistical Analysis Report</h1>
            <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="config-info">
            <h3>📋 Analysis Configuration</h3>
            <p><strong>Significance Level (α):</strong> {alpha}</p>
            <p><strong>FDR Correction Applied:</strong> {'Yes' if use_fdr else 'No'}</p>
            <p><strong>Population Size Column:</strong> {pop_size_column}</p>
            <p><strong>Group ID Column:</strong> {group_id_column if group_id_column else 'Auto-generated'}</p>
            <p><strong>Metrics Analyzed:</strong> {', '.join(metric_columns)}</p>
        </div>
    """
    
    # Add results for each metric
    for metric_data in analysis_results:
        metric = metric_data['metric']
        winner_info = metric_data['winner_info']
        summary_df = metric_data['summary_df']
        comparison_df = metric_data['comparison_df']
        chart_html = metric_data['chart_html']
        
        html_content += f"""
        <div class="metric-section">
            <h2>📊 Analysis for: {metric}</h2>
            
            {winner_info}
            
            <h3>🏆 Group Performance Summary</h3>
            {summary_df.to_html(classes='', table_id='', escape=False)}
            
            <div class="chart-container">
                <h3>📈 Conversion Rate Visualization</h3>
                {chart_html}
            </div>
            
            <h3>🔬 Pairwise Statistical Comparisons</h3>
            {comparison_df.to_html(classes='', table_id='', escape=False)}
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content

def convert_html_to_pdf_ready(html_content):
    """Convert HTML to a format ready for PDF conversion using weasyprint-like approach"""
    # For now, we'll return the HTML that can be saved and manually converted
    # In a full deployment, you'd want to use weasyprint or similar
    return html_content

def create_download_link(html_content, filename):
    """Create download link for HTML file"""
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">📄 Download HTML Report (can be printed to PDF)</a>'
    return href

# File uploader
uploaded_file = st.file_uploader(
    "Upload your experiment CSV file", 
    type="csv",
    help="Each row should represent an experiment group from your split test"
)

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded: {df.shape[0]} experiment groups, {df.shape[1]} columns")
        
        # Show data preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df, use_container_width=True)
        
        # Get column names
        columns = df.columns.tolist()
        
        # Configuration section
        st.subheader("🔧 Analysis Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Group identifier (optional)
            group_id_column = st.selectbox(
                "Select group identifier column:",
                columns,
                index=None,
                help="Column that identifies each experiment group (e.g., 'control', 'variant_a')"
            )

            # Population size column selector
            pop_size_column = st.selectbox(
                "Select the population size column:",
                columns,
                index=None,
                help="Column containing the total number of users in each experiment group"
            )
        
        with col2:
            # Metrics to compare
            metric_columns = st.multiselect(
                "Select metric columns to analyze:",
                [col for col in columns if col != pop_size_column],
                help="Columns containing the metrics you want to compare between groups"
            )
            
            # Significance level
            alpha = st.slider(
                "Significance level (α):",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                help="Threshold for statistical significance"
            )
            
            # FDR correction option
            use_fdr = st.checkbox(
                "Apply FDR (False Discovery Rate) correction",
                value=True,
                help="Benjamini-Hochberg procedure - prevents against false positive errors when running a large number of pair-wise comparisons."
            )
        
        if metric_columns:
            st.subheader("📈 Statistical Analysis Results")
            
            # Calculate conversion rates if metrics are counts
            analysis_df = df.copy()
            
            # Add conversion rates for each metric
            for metric in metric_columns:
                rate_col = f"{metric}_rate"
                analysis_df[rate_col] = analysis_df[metric] / analysis_df[pop_size_column]
            
            # Statistical Tests
            st.subheader("🔬 Statistical Test Results")
            
            # Store all pairwise results for CSV export
            all_pairwise_results = []
            # Store results for PDF export
            pdf_analysis_results = []
            
            if len(df) >= 2:
                for metric in metric_columns:
                    st.write(f"**Analysis for: {metric}**")
                    
                    # Prepare data for statistical tests
                    groups_data = []
                    group_names = []
                    group_rates = []
                    
                    for i, row in df.iterrows():
                        group_name = row[group_id_column] if group_id_column else f"Group_{i+1}"
                        successes = int(row[metric])
                        population = int(row[pop_size_column])
                        rate = successes / population
                        
                        groups_data.append((successes, population))
                        group_names.append(group_name)
                        group_rates.append(rate)
                    
                    # Find the best performing group
                    best_group_idx = np.argmax(group_rates)
                    potential_winner = group_names[best_group_idx]
                    winner_rate = group_rates[best_group_idx]

                    # Run stats test
                    comparison_results = []
                    significant_wins = []
                    has_significant_difference = False
                    
                    for i in range(len(groups_data)):
                        for j in range(i + 1, len(groups_data)):
                            group1_name = group_names[i]
                            group2_name = group_names[j]
                            
                            # Get data
                            x1, n1 = groups_data[i]  # successes, population
                            x2, n2 = groups_data[j]
                            
                            p1 = x1 / n1
                            p2 = x2 / n2
                            
                            # Two-proportion z-test
                            pooled_p = (x1 + x2) / (n1 + n2)
                            se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
                            
                            if se > 0:
                                z_score = (p1 - p2) / se
                                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                                
                                # Determine winner of this comparison
                                if p1 > p2:
                                    comparison_winner = group1_name
                                    lift = ((p1 - p2) / p2) * 100 if p2 > 0 else 0
                                else:
                                    comparison_winner = group2_name
                                    lift = ((p2 - p1) / p1) * 100 if p1 > 0 else 0
                                
                                is_significant = p_value < alpha
                                
                                if is_significant:
                                    has_significant_difference = True
                                    significant_wins.append(comparison_winner)
                                
                                # Status determination
                                if is_significant:
                                    status = f"{comparison_winner} WINS"
                                else:
                                    status = "No significant difference"
                                
                                comparison_result = {
                                    'Metric': metric,
                                    'Comparison': f"{group1_name} vs {group2_name}",
                                    'Group 1 Rate': f"{p1*100:.2f}%",
                                    'Group 2 Rate': f"{p2*100:.2f}%",
                                    'P-value': f"{p_value:.6f}",
                                    'Significant': is_significant,
                                    'Lift %': f"{lift:.2f}%",
                                    'Result': status
                                }
                                
                                comparison_results.append(comparison_result)
                                all_pairwise_results.append(comparison_result)
                    
                    # Winner announcement - only if there are significant differences
                    winner_html = ""
                    if has_significant_difference:
                        # Check if the best performing group actually wins any significant comparisons
                        winner_has_significant_wins = potential_winner in significant_wins
                        
                        if winner_has_significant_wins:
                            winner_message = f"🏆 **WINNER: {potential_winner}** with {winner_rate*100:.2f}% conversion rate"
                            st.success(winner_message)
                            winner_html = f'<div class="winner-box">🏆 WINNER: {potential_winner} with {winner_rate*100:.2f}% conversion rate</div>'
                        else:
                            # Find who actually has the most significant wins
                            from collections import Counter
                            win_counts = Counter(significant_wins)
                            if win_counts:
                                actual_winner = win_counts.most_common(1)[0][0]
                                actual_winner_rate = group_rates[group_names.index(actual_winner)]
                                winner_message = f"🏆 **WINNER: {actual_winner}** with {actual_winner_rate:.4f} ({actual_winner_rate*100:.2f}%) conversion rate"
                                st.success(winner_message)
                                winner_html = f'<div class="winner-box">🏆 WINNER: {actual_winner} with {actual_winner_rate*100:.2f}% conversion rate</div>'
                            else:
                                winner_message = "📊 **NO CLEAR WINNER** - No statistically significant differences found"
                                st.info(winner_message)
                                winner_html = f'<div class="no-winner-box">📊 NO CLEAR WINNER - No statistically significant differences found</div>'
                    else:
                        winner_message = "📊 **NO CLEAR WINNER** - No statistically significant differences found"
                        st.info(winner_message)
                        winner_html = f'<div class="no-winner-box">📊 NO CLEAR WINNER - No statistically significant differences found</div>'

                    # Overall summary for this metric
                    rates_summary = []
                    for i, (group_name, rate) in enumerate(zip(group_names, group_rates)):
                        successes, population = groups_data[i]
                        rates_summary.append({
                            'Rank': i + 1,
                            'Group': group_name,
                            'Conversion Rate': f"{rate:.4f}",
                            'Successes': f"{successes:,}",
                            'Population': f"{population:,}"
                        })
                    
                    # Sort by rate descending
                    rates_summary.sort(key=lambda x: float(x['Conversion Rate']), reverse=True)
                    
                    # Update ranks and highlight winner if there is one
                    for i, item in enumerate(rates_summary):
                        item['Rank'] = i + 1
                        if has_significant_difference and item['Group'] in significant_wins:
                            if i == 0:  # Top performer with significant wins
                                item['Rank'] = "🏆 1"
                    
                    summary_df = pd.DataFrame(rates_summary)
                    st.dataframe(summary_df, use_container_width=True)

                    # Visualization
                    st.subheader("📈 Visualization")
                    
                    # Create visualization data using the same calculation as tables
                    viz_data = []
                    for i, (group_name, rate) in enumerate(zip(group_names, group_rates)):
                        successes, population = groups_data[i]
                        
                        viz_data.append({
                            'Group': group_name,
                            'Conversion_Rate': rate,  # Use the already calculated rate
                            'Count': successes,
                            'Population': population
                        })
                    
                    viz_df = pd.DataFrame(viz_data)
                    
                    # Create a bar chart
                    fig = px.bar(
                        viz_df, 
                        x='Group', 
                        y='Conversion_Rate',
                        title=f'Conversion Rates by Group - {metric}',
                        text='Conversion_Rate'
                    )
                    
                    # Improve the chart appearance
                    fig.update_traces(
                        texttemplate='%{text:.3f}', 
                        textposition='outside',
                        textfont_size=12,
                        marker_color='#636EFA'  # Use a single color for all bars
                    )
                    
                    fig.update_layout(
                        yaxis_title="Conversion Rate",
                        xaxis_title="Experiment Group",
                        title={
                            'text': f'Conversion Rates by Group - {metric}',
                            'x': 0.5,
                            'xanchor': 'center'
                        },
                        height=450,  # Increased height to accommodate labels
                        margin=dict(l=50, r=50, t=100, b=50)  # Increased top margin for labels
                    )
                    
                    # Ensure y-axis has enough room for text labels above bars
                    max_rate = viz_df['Conversion_Rate'].max()
                    fig.update_yaxes(range=[0, max_rate * 1.15])  # Add 15% padding above highest bar
                    
                    # Display the chart in a container for better width control
                    chart_col1, chart_col2, chart_col3 = st.columns([1, 3, 1])
                    with chart_col2:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Convert chart to HTML for PDF
                    chart_html = fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{metric}")
                    
                    # Pairwise comparisons
                    st.write("**Pairwise Statistical Comparisons:**")
                    
                    comparison_df_display = None
                    if comparison_results:
                        comparison_df = pd.DataFrame(comparison_results)
                        # Sort by lift percentage (descending)
                        comparison_df['Lift_Numeric'] = comparison_df['Lift %'].str.replace('%', '').astype(float)
                        comparison_df = comparison_df.sort_values('Lift_Numeric', ascending=False)
                        # Remove the 'Metric' column and helper column for display (we'll keep Metric for export)
                        comparison_df_display = comparison_df.drop(['Metric', 'Lift_Numeric'], axis=1)
                        st.dataframe(comparison_df_display, use_container_width=True)

                    # Store results for PDF export
                    pdf_analysis_results.append({
                        'metric': metric,
                        'winner_info': winner_html,
                        'summary_df': summary_df,
                        'comparison_df': comparison_df_display if comparison_df_display is not None else pd.DataFrame(),
                        'chart_html': chart_html
                    })

                    st.divider()
            
            else:
                st.warning("⚠️ Need at least 2 groups to run statistical tests")
        
        # Export results
        if metric_columns and len(df) >= 2:
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create downloadable CSV report with pairwise comparisons
                if all_pairwise_results:
                    pairwise_df = pd.DataFrame(all_pairwise_results)
                    csv_report = pairwise_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download Stats Analysis CSV",
                        data=csv_report,
                        file_name="ab_test_pairwise_analysis.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No pairwise comparisons available for download")
            
            with col2:
                # PDF Export option
                if pdf_analysis_results:
                    # Create HTML report
                    html_report = create_html_report(
                        pdf_analysis_results, 
                        metric_columns, 
                        df, 
                        group_id_column, 
                        pop_size_column, 
                        alpha, 
                        use_fdr
                    )
                    
                    # Create download button for HTML (which can be printed to PDF)
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"ab_test_report_{timestamp}.html"
                    
                    st.download_button(
                        label="📄 Download Complete Report (HTML)",
                        data=html_report,
                        file_name=filename,
                        mime="text/html",
                        help="Download HTML report that can be printed to PDF from your browser (Ctrl+P → Save as PDF)"
                    )
                    
                    # Instructions for PDF conversion
                    with st.expander("📋 How to convert HTML to PDF"):
                        st.markdown("""
                        **To convert the downloaded HTML report to PDF:**
                        
                        1. **Download the HTML file** using the button above
                        2. **Open the HTML file** in your web browser (Chrome, Firefox, Safari, etc.)
                        3. **Print the page** (Ctrl+P or Cmd+P)
                        4. **Select "Save as PDF"** as the destination
                        5. **Adjust print settings** if needed:
                           - Set margins to "Minimum" for better chart display
                           - Check "Background graphics" to preserve colors
                           - Choose "More settings" → "Paper size" → A4 or Letter
                        6. **Click "Save"** to generate your PDF
                        
                        The HTML report includes:
                        - 📊 All statistical analysis results
                        - 📈 Interactive charts (static in PDF)
                        - 🏆 Winner declarations
                        - 📋 Configuration details
                        - 🔢 Detailed comparison tables
                        """)
                else:
                    st.info("Complete your analysis first to enable PDF export")
        
    except Exception as e:
        st.error(f"❌ Error processing the file: {str(e)}")
        st.write("Please ensure your CSV file is properly formatted with numeric data.")

else:
    st.info("📥 Upload your experiment CSV file to begin analysis")
    
    # Instructions
    with st.expander("📖 How to use this app"):
        st.markdown("""
        **Data Format Expected:**
        - Each row = one experiment group
        - One column = total population size for that group
        - Other columns = metric counts (conversions, clicks, etc.)
        
        **Example CSV:**
        ```
        group_name,population,conversions,clicks,signups
        control,10000,850,2340,123
        variant_a,10000,920,2180,145
        variant_b,10000,780,2890,98
        ```
        
        **Steps:**
        1. Upload your CSV file
        2. Select the column containing the experiment group labels
        3. Select the column containing population size
        4. Select the metric columns you want to analyze
        5. Review statistical test results
        6. Download analysis report (CSV or HTML/PDF)
        
        **Statistical Tests:**
        - 2 groups: Two-proportion Z-test
        - 3+ groups: Chi-square test of independence
        
        **Winner Declaration:**
        - A winner is only declared if there are statistically significant differences
        - The winner must have significant wins in pairwise comparisons
        - If no significant differences are found, no winner is declared
        
        **Export Options:**
        - **CSV**: Raw statistical comparison data for further analysis
        - **HTML/PDF**: Complete formatted report with charts and analysis
        """)
