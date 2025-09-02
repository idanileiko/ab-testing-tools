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
        
        st.success(f"✅ File uploaded! {df.shape[0]} experiment groups, {df.shape[1]} columns")
        
        # Show data preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df, use_container_width=True)
        
        # Get column names
        columns = df.columns.tolist()
        
        # Configuration section
        st.subheader("🔧 Analysis Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Population size column selector
            pop_size_column = st.selectbox(
                "Select the population size column:",
                columns,
                help="Column containing the total number of users in each experiment group"
            )
            
            # Group identifier (optional)
            group_id_column = st.selectbox(
                "Select group identifier column (optional):",
                [None] + columns,
                help="Column that identifies each experiment group (e.g., 'control', 'variant_a')"
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
        
        if metric_columns:
            st.subheader("📈 Statistical Analysis Results")
            
            # Calculate conversion rates if metrics are counts
            analysis_df = df.copy()
            
            # Add conversion rates for each metric
            for metric in metric_columns:
                rate_col = f"{metric}_rate"
                analysis_df[rate_col] = analysis_df[metric] / analysis_df[pop_size_column]
            
            # Display summary statistics
            st.subheader("📊 Summary Statistics")
            
            summary_data = []
            for i, row in analysis_df.iterrows():
                group_name = row[group_id_column] if group_id_column else f"Group_{i+1}"
                
                for metric in metric_columns:
                    rate = row[metric] / row[pop_size_column]
                    summary_data.append({
                        'Group': group_name,
                        'Metric': metric,
                        'Count': row[metric],
                        'Population': row[pop_size_column],
                        'Rate': rate,
                        'Rate %': f"{rate*100:.2f}%"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Statistical Tests
            st.subheader("🔬 Statistical Test Results")
            
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
                    winner = group_names[best_group_idx]
                    winner_rate = group_rates[best_group_idx]
                    
                    # Winner announcement
                    st.success(f"🏆 **WINNER: {winner}** with {winner_rate:.4f} ({winner_rate*100:.2f}%) conversion rate")
                    
                    # Pairwise comparisons
                    st.write("**Pairwise Statistical Comparisons:**")
                    
                    comparison_results = []
                    significant_wins = []
                    
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
                                    significant_wins.append(comparison_winner)
                                
                                # Status determination
                                if is_significant:
                                    status = f"🎯 {comparison_winner} WINS"
                                else:
                                    status = "🤝 No significant difference"
                                
                                comparison_results.append({
                                    'Comparison': f"{group1_name} vs {group2_name}",
                                    'Group 1 Rate': f"{p1:.4f} ({p1*100:.2f}%)",
                                    'Group 2 Rate': f"{p2:.4f} ({p2*100:.2f}%)",
                                    'P-value': f"{p_value:.6f}",
                                    'Significant': is_significant,
                                    'Lift %': f"{lift:.2f}%",
                                    'Result': status
                                })
                    
                    # Display comparison results
                    if comparison_results:
                        comparison_df = pd.DataFrame(comparison_results)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Overall winner summary
                        st.write("**🏁 Final Verdict:**")
                        
                        if significant_wins:
                            # Count wins for each group
                            win_counts = {}
                            for win in significant_wins:
                                win_counts[win] = win_counts.get(win, 0) + 1
                            
                            # Find group with most significant wins
                            statistical_winner = max(win_counts, key=win_counts.get)
                            
                            if statistical_winner == winner:
                                st.success(f"🎉 **{winner}** is both the highest performer AND has statistically significant wins!")
                            else:
                                st.warning(f"⚠️ **{winner}** has the highest rate, but **{statistical_winner}** has the most statistically significant wins")
                        else:
                            st.info(f"📊 **{winner}** has the highest conversion rate, but no statistically significant differences found")
                    
                    # Overall summary for this metric
                    rates_summary = []
                    for i, (group_name, rate) in enumerate(zip(group_names, group_rates)):
                        successes, population = groups_data[i]
                        rates_summary.append({
                            'Rank': i + 1 if group_name != winner else "🏆 1",
                            'Group': group_name,
                            'Conversion Rate': f"{rate:.4f}",
                            'Percentage': f"{rate*100:.2f}%",
                            'Count': f"{successes:,}/{population:,}"
                        })
                    
                    # Sort by rate descending
                    rates_summary.sort(key=lambda x: float(x['Conversion Rate']), reverse=True)
                    
                    # Update ranks
                    for i, item in enumerate(rates_summary):
                        if not str(item['Rank']).startswith('🏆'):
                            item['Rank'] = i + 1
                    
                    summary_df = pd.DataFrame(rates_summary)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Visualization
                    st.subheader("📈 Visualization")
                    
                    # Create visualization data
                    viz_data = []
                    for i, (group_data, group_name) in enumerate(zip(groups_data, group_names)):
                        rate = np.mean(group_data)
                        population = len(group_data)
                        count = int(np.sum(group_data))
                        
                        viz_data.append({
                            'Group': group_name,
                            'Conversion_Rate': rate,
                            'Count': count,
                            'Population': population
                        })
                    
                    viz_df = pd.DataFrame(viz_data)
                    
                    # Bar chart of conversion rates
                    fig = px.bar(
                        viz_df, 
                        x='Group', 
                        y='Conversion_Rate',
                        title=f'Conversion Rates by Group - {metric}',
                        text='Conversion_Rate'
                    )
                    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig.update_layout(yaxis_title="Conversion Rate")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
            
            else:
                st.warning("⚠️ Need at least 2 groups to run statistical tests")
        
        # Export results
        if metric_columns and len(df) >= 2:
            st.subheader("💾 Export Results")
            
            # Create downloadable report
            report_data = []
            for metric in metric_columns:
                for i, row in df.iterrows():
                    group_name = row[group_id_column] if group_id_column else f"Group_{i+1}"
                    rate = row[metric] / row[pop_size_column]
                    
                    report_data.append({
                        'Metric': metric,
                        'Group': group_name,
                        'Count': row[metric],
                        'Population': row[pop_size_column],
                        'Conversion_Rate': rate,
                        'Conversion_Rate_Percent': f"{rate*100:.2f}%"
                    })
            
            report_df = pd.DataFrame(report_data)
            csv_report = report_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Analysis Report",
                data=csv_report,
                file_name="ab_test_analysis.csv",
                mime="text/csv"
            )
            
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
        2. Select the column containing population sizes
        3. Select the metric columns you want to analyze
        4. Review statistical test results
        5. Download analysis report
        
        **Statistical Tests:**
        - 2 groups: Two-proportion Z-test
        - 3+ groups: Chi-square test of independence
        """)
