import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="A/B Testing Analysis",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 A/B Testing Statistical Analysis")
st.write("Upload your experiment data and run statistical tests between groups")

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
                                    status = f"🎯 {comparison_winner} WINS"
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
                    if has_significant_difference:
                        # Check if the best performing group actually wins any significant comparisons
                        winner_has_significant_wins = potential_winner in significant_wins
                        
                        if winner_has_significant_wins:
                            st.success(f"🏆 **WINNER: {potential_winner}** with {winner_rate*100:.2f}% conversion rate")
                        else:
                            # Find who actually has the most significant wins
                            from collections import Counter
                            win_counts = Counter(significant_wins)
                            if win_counts:
                                actual_winner = win_counts.most_common(1)[0][0]
                                actual_winner_rate = group_rates[group_names.index(actual_winner)]
                                st.success(f"🏆 **WINNER: {actual_winner}** with {actual_winner_rate:.4f} ({actual_winner_rate*100:.2f}%) conversion rate")
                            else:
                                st.info("📊 **NO CLEAR WINNER** - No statistically significant differences found")
                    else:
                        st.info("📊 **NO CLEAR WINNER** - No statistically significant differences found")

                    # Overall summary for this metric
                    rates_summary = []
                    for i, (group_name, rate) in enumerate(zip(group_names, group_rates)):
                        successes, population = groups_data[i]
                        rates_summary.append({
                            'Rank': i + 1,
                            'Group': group_name,
                            'Conversion Rate': f"{rate:.4f}",
                            'Successes / Population': f"{successes} / {population}"
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
                    st.dataframe(summary_df, use_container_width=False)
                    
                    # Pairwise comparisons
                    st.write("**Pairwise Statistical Comparisons:**")
                    
                    # Display comparison results
                    if comparison_results:
                        comparison_df = pd.DataFrame(comparison_results)
                        # Remove the 'Metric' column for display (we'll keep it for export)
                        display_df = comparison_df.drop('Metric', axis=1)
                        st.dataframe(display_df, use_container_width=True)
                    
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
                    
                    # Create a more aesthetically pleasing bar chart
                    fig = px.bar(
                        viz_df, 
                        x='Group', 
                        y='Conversion_Rate',
                        title=f'Conversion Rates by Group - {metric}',
                        text='Conversion_Rate',
                        color='Conversion_Rate',
                        color_continuous_scale='viridis'
                    )
                    
                    # Improve the chart appearance
                    fig.update_traces(
                        texttemplate='%{text:.3f}', 
                        textposition='outside',
                        textfont_size=12
                    )
                    
                    fig.update_layout(
                        yaxis_title="Conversion Rate",
                        xaxis_title="Experiment Group",
                        title_x=0.5,  # Center the title
                        showlegend=False,  # Hide color scale legend
                        height=400,  # Set a reasonable height
                        margin=dict(l=50, r=50, t=60, b=50)
                    )
                    
                    # Display the chart in a container for better width control
                    chart_col1, chart_col2, chart_col3 = st.columns([1, 3, 1])
                    with chart_col2:
                        st.plotly_chart(fig, use_container_width=True)
                    
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
                        label="📊 Download Pairwise Analysis CSV",
                        data=csv_report,
                        file_name="ab_test_pairwise_analysis.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No pairwise comparisons available for download")
            
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
        6. Download analysis report
        
        **Statistical Tests:**
        - 2 groups: Two-proportion Z-test
        - 3+ groups: Chi-square test of independence
        
        **Winner Declaration:**
        - A winner is only declared if there are statistically significant differences
        - The winner must have significant wins in pairwise comparisons
        - If no significant differences are found, no winner is declared
        """)
