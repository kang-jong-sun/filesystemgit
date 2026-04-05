const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageNumber, PageBreak, LevelFormat } = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellM = { top: 60, bottom: 60, left: 80, right: 80 };

function cell(text, w, opts = {}) {
    const shading = opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined;
    return new TableCell({
        borders, width: { size: w, type: WidthType.DXA }, margins: cellM, shading,
        children: [new Paragraph({
            alignment: opts.align || AlignmentType.CENTER,
            children: [new TextRun({ text: String(text), bold: opts.bold, size: opts.size || 16, font: "Arial" })]
        })]
    });
}

// Data
const returnBest = [
    {rank:1,ver:'v22.2',file:'v22.2.md',ret:'+1,036,934%',amt:'$31,111,013',y20:'17T',y21:'16T',y22:'17T',y23:'20T',y24:'15T',y25:'9T',y26:'3T',total:'97',yMDD:'71.3%',yPF:'1.35',eng:'6/6',note:'60%마진, 최고수익'},
    {rank:2,ver:'v22.3',file:'v22.3.md',ret:'+555,704%',amt:'$16,674,108',y20:'32T',y21:'18T',y22:'24T',y23:'35T',y24:'31T',y25:'22T',y26:'6T',total:'168',yMDD:'74.6%',yPF:'1.54',eng:'5/6',note:'WMA250 Slow MA'},
    {rank:3,ver:'v32.2',file:'v32.2.md',ret:'+481,367%',amt:'$24,073,329',y20:'12T',y21:'11T',y22:'12T',y23:'14T',y24:'10T',y25:'8T',y26:'3T',total:'70',yMDD:'43.5%',yPF:'5.8',eng:'6/6 (4엔진검증완료)',note:'TSL->SL비활성, PF5.8'},
    {rank:4,ver:'v15.4',file:'v15.4.md',ret:'+290,489%',amt:'$8,717,659',y20:'17T',y21:'16T',y22:'17T',y23:'20T',y24:'15T',y25:'13T',y26:'7T',total:'105',yMDD:'54.2%',yPF:'1.65',eng:'6/6',note:'40%마진 복리폭발'},
    {rank:5,ver:'v32.3',file:'v32.3.md',ret:'+264,631%',amt:'$13,236,537',y20:'12T',y21:'11T',y22:'13T',y23:'16T',y24:'8T',y25:'8T',y26:'1T',total:'69',yMDD:'43.5%',yPF:'3.2',eng:'6/6 (4엔진검증완료)',note:'EMA75/SMA750, 낮은MDD'},
    {rank:6,ver:'v25.1A',file:'v25.1A.md',ret:'+65,695%',amt:'$1,973,859',y20:'7T',y21:'5T',y22:'5T',y23:'9T',y24:'10T',y25:'6T',y26:'3T',total:'45',yMDD:'46.2%',yPF:'62.4',eng:'5/6',note:'HMA21/EMA250, PF62'},
    {rank:7,ver:'v24.2',file:'v24.2.md',ret:'+58,566%',amt:'$1,759,977',y20:'7T',y21:'7T',y22:'6T',y23:'8T',y24:'7T',y25:'4T',y26:'2T',total:'41',yMDD:'83.8%',yPF:'10.5',eng:'5/6',note:'1h봉, 70%마진'},
    {rank:8,ver:'v15.5',file:'v15.5.md',ret:'+55,887%',amt:'$1,679,610',y20:'14T',y21:'12T',y22:'13T',y23:'17T',y24:'13T',y25:'10T',y26:'4T',total:'83',yMDD:'39.8%',yPF:'2.52',eng:'6/6',note:'손실연도 0, 가장 일관'},
    {rank:9,ver:'v23.4',file:'v23.4.md',ret:'+48,650%',amt:'$2,437,498',y20:'18T',y21:'16T',y22:'17T',y23:'20T',y24:'17T',y25:'14T',y26:'7T',total:'109',yMDD:'44.1%',yPF:'1.89',eng:'6/6',note:'고빈도 109거래'},
    {rank:10,ver:'v15.6_2',file:'v15.6_2.md',ret:'+42,620%',amt:'$1,281,597',y20:'14T',y21:'12T',y22:'13T',y23:'17T',y24:'14T',y25:'12T',y26:'4T',total:'86',yMDD:'48.8%',yPF:'2.52',eng:'6/6',note:'손실연도 0, Model B'},
];

const stabilityBest = [
    {rank:1,ver:'v16.6',file:'v16.6.md',ret:'+1,144%',amt:'$37,318',y20:'6T',y21:'5T',y22:'5T',y23:'6T',y24:'5T',y25:'3T',y26:'2T',total:'32',yMDD:'10.1%',yPF:'14.2',eng:'6/6',note:'SL 0, MDD 10%, 손실연0'},
    {rank:2,ver:'v17.0_4',file:'v17.0_4.md',ret:'+1,144%',amt:'$37,318',y20:'6T',y21:'5T',y22:'5T',y23:'6T',y24:'5T',y25:'3T',y26:'2T',total:'32',yMDD:'10.1%',yPF:'14.2',eng:'6/6',note:'v16.6+ML보호, 손실연0'},
    {rank:3,ver:'v16.2_2',file:'v16.2_2.md',ret:'+1,839%',amt:'$58,179',y20:'7T',y21:'7T',y22:'7T',y23:'8T',y24:'8T',y25:'5T',y26:'2T',total:'44',yMDD:'8.1%',yPF:'12.9',eng:'6/6',note:'SL 0, 역대최저MDD 8.1%'},
    {rank:4,ver:'v16.4',file:'v16.4.md',ret:'+1,839%',amt:'$58,179',y20:'7T',y21:'7T',y22:'7T',y23:'8T',y24:'8T',y25:'5T',y26:'2T',total:'44',yMDD:'8.1%',yPF:'12.9',eng:'6/6',note:'ADX Wilder수정, SL 0'},
    {rank:5,ver:'v22.0_2',file:'v22.0_2.md',ret:'+4,414%',amt:'$135,427',y20:'8T',y21:'6T',y22:'6T',y23:'9T',y24:'8T',y25:'5T',y26:'3T',total:'45',yMDD:'13.3%',yPF:'16.9',eng:'6/6',note:'SL 0, 손실연 0'},
    {rank:6,ver:'v22.1_2',file:'v22.1_2.md',ret:'+4,321%',amt:'$132,634',y20:'7T',y21:'5T',y22:'5T',y23:'8T',y24:'8T',y25:'5T',y26:'3T',total:'41',yMDD:'13.3%',yPF:'16.8',eng:'6/6',note:'SL 0, 손실연 0'},
    {rank:7,ver:'v25.1A',file:'v25.1A.md',ret:'+65,695%',amt:'$1,973,859',y20:'7T',y21:'5T',y22:'5T',y23:'9T',y24:'10T',y25:'6T',y26:'3T',total:'45',yMDD:'46.2%',yPF:'62.4',eng:'5/6',note:'PF 62.4 최고'},
    {rank:8,ver:'v16.0',file:'v16.0.md',ret:'+9,367%',amt:'$284,021',y20:'7T',y21:'7T',y22:'7T',y23:'8T',y24:'8T',y25:'5T',y26:'2T',total:'44',yMDD:'13.3%',yPF:'11.9',eng:'6/6',note:'WMA3 최초발견, SL 0'},
    {rank:9,ver:'v32.2',file:'v32.2.md',ret:'+481,367%',amt:'$24,073,329',y20:'12T',y21:'11T',y22:'12T',y23:'14T',y24:'10T',y25:'8T',y26:'3T',total:'70',yMDD:'43.5%',yPF:'5.8',eng:'6/6',note:'수익+안정 동시'},
    {rank:10,ver:'v17.0_2',file:'v17.0_2.md',ret:'+2,809%',amt:'$87,261',y20:'7T',y21:'6T',y22:'6T',y23:'8T',y24:'7T',y25:'5T',y26:'2T',total:'41',yMDD:'27.4%',yPF:'11.1',eng:'6/6',note:'SL 2, Core엔진'},
];

const discardBest = [
    {rank:1,ver:'v28.0_2',file:'v28.0_2.md',ret:'-100%',amt:'$6',y20:'-',y21:'-',y22:'-',y23:'-',y24:'-',y25:'-',y26:'-',total:'86',yMDD:'99.8%',yPF:'0.56',eng:'0/6',note:'파산, LIQ 4회'},
    {rank:2,ver:'v23.2',file:'v23.2.md',ret:'-100%',amt:'$10',y20:'-',y21:'-',y22:'-',y23:'-',y24:'-',y25:'-',y26:'-',total:'193',yMDD:'99.9%',yPF:'0.88',eng:'0/6',note:'파산, LIQ 3회'},
    {rank:3,ver:'v28.0_3',file:'v28.0_3.md',ret:'-88%',amt:'$373',y20:'-',y21:'-',y22:'-',y23:'-',y24:'-',y25:'-',y26:'-',total:'46',yMDD:'91.3%',yPF:'0.64',eng:'0/6',note:'대손실, LIQ 2회'},
    {rank:4,ver:'v26.0_1',file:'v26.0_1.md',ret:'-66%',amt:'$1,018',y20:'-',y21:'-',y22:'-',y23:'-',y24:'-',y25:'-',y26:'-',total:'84',yMDD:'91.3%',yPF:'0.90',eng:'0/6',note:'손실, MDD 91%'},
    {rank:5,ver:'v27_2',file:'v27_2.md',ret:'-72%',amt:'$832',y20:'-',y21:'-',y22:'-',y23:'-',y24:'-',y25:'-',y26:'-',total:'55',yMDD:'84.0%',yPF:'0.74',eng:'1/6',note:'LIQ 22회'},
    {rank:6,ver:'v27_1',file:'v27_1.md',ret:'-67%',amt:'$997',y20:'-',y21:'-',y22:'-',y23:'-',y24:'-',y25:'-',y26:'-',total:'53',yMDD:'76.9%',yPF:'0.81',eng:'2/6',note:'LIQ 17회'},
    {rank:7,ver:'v25.2_3',file:'v25.2_3.md',ret:'-30%',amt:'$2,096',y20:'-',y21:'-',y22:'-',y23:'-',y24:'-',y25:'-',y26:'-',total:'54',yMDD:'83.1%',yPF:'0.90',eng:'2/6',note:'손실, LIQ 2회'},
    {rank:8,ver:'v25.1',file:'v25.1.md',ret:'-42%',amt:'$1,734',y20:'-',y21:'-',y22:'-',y23:'-',y24:'-',y25:'-',y26:'-',total:'34',yMDD:'85.0%',yPF:'0.82',eng:'1/6',note:'손실, LIQ 2회'},
    {rank:9,ver:'v22.8',file:'v22.8.md',ret:'-49%',amt:'$2,529',y20:'-',y21:'-',y22:'-',y23:'-',y24:'-',y25:'-',y26:'-',total:'78',yMDD:'89.4%',yPF:'0.93',eng:'2/6',note:'장기교차 실패'},
    {rank:10,ver:'v16.2F_1',file:'v16.2F_1.md',ret:'-27%',amt:'$2,190',y20:'-',y21:'-',y22:'-',y23:'-',y24:'-',y25:'-',y26:'-',total:'53',yMDD:'75.2%',yPF:'0.92',eng:'2/6',note:'LIQ 3회'},
];

function makeTable(data, title) {
    const cols = [500,1200,1100,1300,550,550,550,550,550,550,550,600,700,700,900,1100];
    const headers = ['순위','파일명','손익률','손익금액','20','21','22','23','24','25','26','총거래','MDD','PF','6엔진일치','비고(사유)'];

    const headerRow = new TableRow({
        children: headers.map((h,i) => cell(h, cols[i], { bold: true, fill: "2E75B6", size: 14 }))
    });

    const rows = data.map(d => new TableRow({
        children: [
            cell(d.rank, cols[0], {size:14}),
            cell(d.ver, cols[1], {size:13, bold:true}),
            cell(d.ret, cols[2], {size:13}),
            cell(d.amt, cols[3], {size:13}),
            cell(d.y20, cols[4], {size:12}),
            cell(d.y21, cols[5], {size:12}),
            cell(d.y22, cols[6], {size:12}),
            cell(d.y23, cols[7], {size:12}),
            cell(d.y24, cols[8], {size:12}),
            cell(d.y25, cols[9], {size:12}),
            cell(d.y26, cols[10], {size:12}),
            cell(d.total, cols[11], {size:13, bold:true}),
            cell(d.yMDD, cols[12], {size:13}),
            cell(d.yPF, cols[13], {size:13}),
            cell(d.eng, cols[14], {size:12}),
            cell(d.note, cols[15], {size:11, align: AlignmentType.LEFT}),
        ]
    }));

    return [
        new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: {before:400,after:200},
            children: [new TextRun({ text: title, font: "Arial" })] }),
        new Table({ width: { size: 14400, type: WidthType.DXA }, columnWidths: cols, rows: [headerRow, ...rows] }),
    ];
}

const doc = new Document({
    styles: {
        default: { document: { run: { font: "Arial", size: 22 } } },
        paragraphStyles: [
            { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 36, bold: true, font: "Arial", color: "1F4E79" },
              paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 } },
            { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 28, bold: true, font: "Arial", color: "2E75B6" },
              paragraph: { spacing: { before: 280, after: 180 }, outlineLevel: 1 } },
        ]
    },
    sections: [{
        properties: {
            page: {
                size: { width: 15840, height: 12240, orientation: "landscape" },
                margin: { top: 720, right: 720, bottom: 720, left: 720 }
            }
        },
        headers: {
            default: new Header({ children: [new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [new TextRun({ text: "BTC/USDT Futures Trading System - 46 Planning Documents x 6 Engine Cross Verification Report", size: 16, color: "888888", font: "Arial" })]
            })] })
        },
        footers: {
            default: new Footer({ children: [new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [new TextRun({ text: "Page ", size: 16, font: "Arial" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, font: "Arial" })]
            })] })
        },
        children: [
            // Title
            new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER,
                children: [new TextRun({ text: "BTC/USDT Futures Trading System", font: "Arial" })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: {after: 100},
                children: [new TextRun({ text: "46 Planning Documents x 6 Engine Cross Verification Final Report", size: 28, font: "Arial", color: "2E75B6" })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: {after: 400},
                children: [new TextRun({ text: "Date: 2026-04-01 | Data: 75 months (2020.01~2026.03) | 655,399 5m candles", size: 20, font: "Arial", color: "666666" })] }),

            // Summary
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "Executive Summary", font: "Arial" })] }),
            new Paragraph({ spacing: {after:100}, children: [new TextRun({ text: "46 planning documents verified with 6 engines: E1(Standard), E2(ADX-5), E3(ADX+5), E4(RSI widened), E5(Margin-10%), E6(SL+2%)", size: 20, font: "Arial" })] }),
            new Paragraph({ spacing: {after:200}, children: [new TextRun({ text: "Engines confirmed: bt_fast (ewm ADX), parameter sensitivity (ADX/RSI/Margin/SL variations), v32.2/v32.3 verified with 4 additional independent engines (Numba JIT, Pure Python, Numpy State, Class OOP)", size: 20, font: "Arial" })] }),

            new PageBreak(),
            // Return BEST 10
            ...makeTable(returnBest, "1. Return BEST 10"),
            new PageBreak(),
            // Stability BEST 10
            ...makeTable(stabilityBest, "2. Stability BEST 10"),
            new PageBreak(),
            // Discard BEST 10
            ...makeTable(discardBest, "3. Discard BEST 10"),

            new PageBreak(),
            // Conclusion
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: "4. Conclusion & Recommendations", font: "Arial" })] }),
            new Paragraph({ spacing: {after:100}, children: [new TextRun({ text: "Top Recommended: v32.2 (Return #3 + Stability #9 = Best Overall)", size: 22, bold: true, font: "Arial" })] }),
            new Paragraph({ spacing: {after:100}, children: [new TextRun({ text: "- $5,000 -> $24,073,329 (+481,367%) | PF 5.8 | MDD 43.5% | 70 trades | 4-engine verified", size: 20, font: "Arial" })] }),
            new Paragraph({ spacing: {after:100}, children: [new TextRun({ text: "Safest: v16.4/v16.2_2 (MDD 8.1%, PF 12.9, SL 0)", size: 22, bold: true, font: "Arial" })] }),
            new Paragraph({ spacing: {after:100}, children: [new TextRun({ text: "Most Consistent: v15.5 (0 loss years in 7 years, 6/6 engines profitable)", size: 22, bold: true, font: "Arial" })] }),
            new Paragraph({ spacing: {after:100}, children: [new TextRun({ text: "Discard: 10 versions with negative returns and 0-2/6 engine profitability", size: 22, bold: true, font: "Arial", color: "CC0000" })] }),
        ]
    }]
});

Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("D:\\filesystem\\futures\\btc_V1\\test4\\BTC_Trading_System_Final_Report.docx", buffer);
    console.log("Report saved: BTC_Trading_System_Final_Report.docx");
});
