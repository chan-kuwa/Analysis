import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import os
from io import BytesIO

# --- ページ設定 ---
# 画像ファイルのパスを設定（スクリプトと同じディレクトリにある想定）
icon_path = "hro.png"

# 画像ファイルが存在する場合のみ設定を適用
if os.path.exists(icon_path):
    st.set_page_config(
        page_title="Risk Analysis Engine",
        page_icon=icon_path, # ブラウザのタブアイコンを設定
        layout="wide"
    )
else:
    # 画像がない場合のフォールバック（デフォルトの挙動）
    st.set_page_config(
        page_title="Risk Analysis Engine",
        layout="wide"
    )
    st.warning(f"警告: アイコン画像ファイル '{icon_path}' が見つかりません。タイトル横の画像は表示されません。")


# --- タイトル部分（画像とテキストを横並びにする） ---
if os.path.exists(icon_path):
    # 2つのカラムを作成。1つ目は画像用、2つ目はタイトル用。
    # 幅の比率を 1:20 程度にして、画像を小さく表示させる。
    col1, col2 = st.columns([1, 20])
    
    with col1:
        # タイトルの高さに合わせるため、少しパディング調整（必要に応じてuse_container_widthなどを調整）
        st.image(icon_path, width=45) # widthで画像の大きさを微調整

    with col2:
        # カラム2にメインタイトルを表示
        st.title("Risk Structure Analysis Engine")
else:
    # 画像がない場合は通常通りタイトルのみ表示
    st.title("🔬 Risk Structure Analysis Engine")


# --- キャプション ---
st.caption("評価データ(CSV)を統合し、HRO原則に基づいた多職種解析を実行します")

# --- サイドバー：API設定 & マスター定義 ---
with st.sidebar:
    st.header("🔑 システム設定")
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        st.success("API認証済み")
    
    
    selected_model = "gemini-3-flash-preview"

    st.divider()
    st.header("⚙️ 評価定義 (分析用参照)")
    # 元のコードの定義値を保持
    sev_def = {3: "発生時に試験結果の信頼性、安全性、被験者保護に即時の影響を与える", 
               2: "蓄積して発生することで影響を与える", 1: "ほとんど影響を与えない"}
    occ_def = {3: "繰り返し発生し蓄積する", 2: "発生は偶発的", 1: "ほとんど発生しない"}
    det_def = {1: "現場で即時検出可能", 2: "データで検出可能", 3: "訪問モニタリングで検出可能"}

# --- メインコンテンツ ---
st.header("1. 評価データのインポート")
uploaded_files = st.file_uploader("Risk Assessment Tool Proで出力したCSVを複数選択してください", 
                                  type="csv", accept_multiple_files=True)

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        dfs.append(pd.read_csv(file, encoding='utf-8-sig'))
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    st.subheader("📊 統合データプレビュー")
    st.dataframe(combined_df, use_container_width=True)

    st.divider()
    st.header("2. 分析対象の選択と可視化")
    
    # CSV内のリスク事象からセレクトボックスで選択
    target_risks = combined_df["risk_event"].unique()
    selected_risk = st.selectbox("分析するリスク事象を選択", target_risks)
    
    analysis_data = combined_df[combined_df["risk_event"] == selected_risk]

    # --- 可視化セクション ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("職種別 S/O/D スコア")
        # 職種ごとのS, O, Dを比較するレーダーチャート
        fig_radar = go.Figure()
        for _, row in analysis_data.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['S'], row['O'], row['D']],
                theta=['S (Severity)', 'O (Occurrence)', 'D (Detectability)'],
                fill='toself',
                name=row['role']
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 3])))
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.subheader("RPN (リスク優先度) 比較")
        analysis_data['RPN'] = analysis_data['S'] * analysis_data['O'] * analysis_data['D']
        fig_bar = px.bar(analysis_data, x='role', y='RPN', color='role', text_auto=True)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 解析実行 ---
    st.divider()
    if st.button("🚀 構造的差異の解析を実行"):
        if not api_key:
            st.error("APIキーが必要です")
        else:
            try:
                model = genai.GenerativeModel(selected_model)
                
                # 元のプロンプトを完全保持 + 言い換え指示のみ追加
                prompt = f"""
あなたは「多職種評価データの構造分析エンジン」である。
以下の評価データを分析せよ。

【リスク内容】
{selected_risk}

【評価定義】
- 影響度(S): 3={sev_def[3]}, 2={sev_def[2]}, 1={sev_def[1]}
- 発生頻度(O): 3={occ_def[3]}, 2={occ_def[2]}, 1={occ_def[1]}
- 検出性(D): 1={det_def[1]}, 2={det_def[2]}, 3={det_def[3]}

【評価データ】
{analysis_data.to_dict(orient='records')}

入力データに対し、以下の3段階で出力せよ。

---
【STEP1：構造抽出（事実のみ）】
以下を記述：
・影響度, 発生頻度, 検出性の分布（値・平均・分散・一致/不一致）
・職種の役割ごとの差異（大小関係）
・因子分布（集中/分散と内訳）

禁止：意味付け、原因推定、一般知識、新規概念語

---
【STEP2：制約付き推論】
以下のテンプレのみ使用：
1. 勾配型：「{{指標}}は{{対象A}}から{{対象B}}にかけて{{増加/減少}}する」
2. 分離型：「{{指標}}は{{グループA}}と{{グループB}}で値が分離している」
3. 合意型：「{{指標}}は全職種で一致している（値：X）」
4. 因子分布型：「因子は{{集中/分散}}しており、{{内訳}}で構成される」

ルール：テンプレ厳守、因果関係・評価・意見の禁止。

---
【STEP3：パターン選択】
以下から最も近いものを1つ選択：
A：工程依存型（勾配＋因子分散）
B：認識乖離型（分離＋因子集中）
C：安定型（全一致）
D：該当なし

理由は完結に記述せよ。

【ステップ4：リスク特性の記述】
ステップ2、ステップ3で得られた内容と、定義されたリスク指標および職種の役割、選択された因子の組み合わせから論理的に得られたリスク特性を記述すること。
ルール:得られた結果から直接導かれる内容のみを記述すること。

【ステップ5】リスク低減策の方向性と議論のポイント
優秀なファシリテーターとしてステップ4のリスク特性を基にリスクの効果的な低減を目的としてどういった議論が必要であるのか記述すること。
ルール:一般的な知識や経験に基づく内容の記述は禁止し、ステップ4のリスク特性から直接導かれる内容のみを記述すること。得られた結果ごとに改行して記述すること。

---
【まとめ：本リスクの分析結果の要約】
上記STEP1〜5の結果を、専門外のステークホルダーでも理解できるよう、日常的な平易な言葉で言い換えなさい。
ただし、新たな意味の付与や推測、感情的な装飾は厳禁とし、分析結果から導かれた論理的帰結の維持に留めること。

---
【出力形式】
・STEPごとに見出しをつける
・1文章ごとに改行をする
【HROに基づく制約】
以下を厳守せよ：
・単一の原因に還元しない（Reluctance to Simplify）
・最も高いリスク評価を無視しない（Preoccupation with Failure）
・現場に近い職種の評価を軽視しない（Sensitivity to Operations）
・最も関連性の高い職種の評価を優先して扱う（Deference to Expertise）
・差異を解消するのではなく、差異の存在を前提に記述する

禁止：
・「主な原因は〜である」と断定すること
・複数因子を1つに統合すること
・平均値のみで結論を出すこと
"""
                with st.spinner("AIが構造を解析中..."):
                    response = model.generate_content(prompt)
                    st.divider()
                    st.subheader("🤖 AI構造分析レポート")
                    st.markdown(response.text)
                    
                    # 結果の保存機能（テキストファイル出力）
                    report_text = response.text
                    st.download_button(
                        label="📥 分析結果を保存 (Text)",
                        data=report_text,
                        file_name=f"Risk_Analysis_{selected_risk}.txt",
                        mime="text/plain"
                    )

            except Exception as e:
                st.error(f"解析エラー: {e}")
else:
    st.info("Risk Assessment Tool Proから出力されたCSVファイルをアップロードしてください。")
